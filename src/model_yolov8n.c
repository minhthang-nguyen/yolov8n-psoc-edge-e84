/*
 * Adapted from model_yolov5n.c to run YOLOv8n on Infineon Edge E84
 *
 * Key differences vs YOLOv5n:
 *   - Input  : uint8_t[320,320,3] OR float[320,320,3]  <- chọn bằng #define bên dưới
 *   - Output : float[8, 2100] raw (không NMS, không detect_flag)
 *              channels: { x, y, w, h, ThanLa, RaNu, NoHoa, Harvest }
 *              (YOLOv8 không có object-confidence channel)
 *   - Dequant: scale=0.008074194192886353, zero_point=-112
 *   - Memory : Buffer=324000, State=3101712, Readonly=2808352
 */

#include "inference_task.h"

#ifndef COMPONENT_ML_TFLM
    #error Symbol COMPONENT_ML_TFLM is not defined. Add 'COMPONENTS+=ML_TFLM' to your Makefile.
#endif

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "cy_retarget_io.h"
#include "cy_utils.h"
#include "mtb_ml_model.h"
#include "mtb_ml_utils.h"
#include "mtb_ml.h"

#include "model.h"   // binary của YOLOv8n — thay "model.h" của YOLOv5n


#ifdef __GNUC__
    #define ALIGNED(x) __attribute__((aligned(x)))
#else
    #define ALIGNED(x) __declspec(align(x))
#endif

/* ------------------------------------------------------------------ */
/* Memory sizes                                                        */
/* ------------------------------------------------------------------ */
#define BUFFER_SIZE     324000
#define STATE_SIZE      3101712
#define READONLY_SIZE   2808352

/* ------------------------------------------------------------------ */
/* YOLOv8 output dimensions                                            */
/*   Raw model output: int8[8, 2100] = 16800 bytes                    */
/*   After dequant   : float[8, 2100] = 67200 bytes                   */
/* ------------------------------------------------------------------ */
#define YOLOv8_CHANNELS     8       // x, y, w, h, cls0..cls3
#define YOLOv8_DETECTIONS   2100
#define YOLOv8_RAW_BYTES    (YOLOv8_CHANNELS * YOLOv8_DETECTIONS)  // 16800

/* Quantization params (từ model_yolov8n.c gốc của Imagimob) */
#define YOLOv8_DEQUANT_SCALE    0.008074194192886353f
#define YOLOv8_DEQUANT_OFFSET   (-112)

/* ------------------------------------------------------------------ */
/* Static memory — đặt trong SOCMEM giống YOLOv5n                     */
/* ------------------------------------------------------------------ */
static __attribute__((section(".cy_socmem_data"), aligned(16))) int8_t _buffer[BUFFER_SIZE];
static __attribute__((section(".cy_socmem_data"), aligned(16))) int8_t _state[STATE_SIZE];

/* Model weights trong flash */
static const __attribute__((aligned(16))) uint32_t _K3_flash[] = {
    
};

/* ------------------------------------------------------------------ */
/* Memory-mapped aliases                                               */
/*                                                                     */
/*  _K2 = tensor arena  (state+0x10,   3101696 bytes)                 */
/*  _K6 = model handle  (state+0x00,   8 bytes — ptr to mtb_ml_model) */
/*  _K3 = model weights (flash,         2808352 bytes)                 */
/*                                                                     */
/*  Buffer layout:                                                     */
/*   [0x00000 .. 0x041A0) = 16800 B  -> _K5 int8 raw output           */
/*                                      (= _K7 float output,          */
/*                                        dequant dùng chung vùng này)*/
/*   [0x041A0 .. 0xF0060) = 307200 B -> _K1 int8 input scratch        */
/*                                                                     */
/*  Không overlap vì 16800 < 0x41A0 (16800 bytes đúng bằng 0x41A0)   */
/* ------------------------------------------------------------------ */
#define _K3     ((uint8_t *)_K3_flash)
#define _K2     ((uint8_t *)(_state  + 0x00000010))  // tensor arena
#define _K6     ((int8_t  *)(_state  + 0x00000000))  // model handle ptr (8 bytes)
#define _K1     ((int8_t  *)(_buffer + 0x00000000))  // input scratch   (307200 bytes)
#define _K5     ((int8_t  *)(_buffer + 0x0004b000))  // raw int8 output  (16800 bytes)
#define _K7     ((float   *)(_buffer + 0x00000000))  // float output     (67200 bytes)
                                                     // NOTE: _K5 & _K7 cùng địa chỉ —
                                                     // dequant ghi đè sau khi inference xong

#define IPWIN_RET_SUCCESS    0
#define IPWIN_RET_NODATA    -1
#define IPWIN_RET_ERROR     -2
#define IPWIN_RET_STREAMEND -3

/* ====================================================================
 * Input helpers
 * ==================================================================== */

/* Path A: uint8_t [0,255] -> int8 [-128,127] (range shift) */
static inline void cast_u8_to_i8(const uint8_t *restrict in,
                                   int8_t        *restrict out,
                                   int32_t count)
{
    for (int i = 0; i < count; i++)
        out[i] = (int8_t)((int)in[i] - 128);
}

static inline void cast_u8_to_i8_yolov8(
    const void *handle,
    const uint8_t *restrict input,
    int8_t *restrict output,
    int32_t count)
{
    mtb_ml_model_t *model = *(mtb_ml_model_t **)handle;

    int in_offset = model->input_zero_point;
    float in_scale = model->input_scale;

    for (int i = 0; i < count; i++)
    {
        // uint8 -> float [0..1]
        float src_float = input[i] / 255.0f;

        // quantize
        float value = (src_float / in_scale) + in_offset;

        if (value > 127.0f)
            output[i] = 127;
        else if (value < -128.0f)
            output[i] = -128;
        else
            output[i] = (int8_t)value;
    }
}

static inline void mtb_model_uint8_to_quantized_int8_like_yolov8(
    const void *handle,
    const uint8_t *restrict src,
    int src_count,
    void *restrict dst,
    int dst_count,
    int8_t *tmp)
{
    mtb_ml_model_t *model = *(mtb_ml_model_t **)handle;

    int in_offset = model->input_zero_point;
    float in_scale = model->input_scale;
	printf("input_scale = %f\r\n", model->input_scale);
	printf("input_zero_point = %d\r\n", model->input_zero_point);
	printf("output_scale = %f\r\n", model->output_scale);
	printf("output_zero_point = %d\r\n", model->output_zero_point);	

    for (int i = 0; i < src_count; i++) {
        float src_float = src[i] / 255.0f;
        float value = (src_float / in_scale) + in_offset;

        if (value > 127.0f) {
            tmp[i] = 127;
        } else if (value < -128.0f) {
            tmp[i] = -128;
        } else {
            tmp[i] = (int8_t)value;
        }
    }

    mtb_ml_model_run(model, (MTB_ML_DATA_T *)tmp);
    //printf("mtb_ml_model_run done \n");
    memcpy(dst, model->output, dst_count);
    //printf("memcpy done \n");
}

/* Path B: float [0.0,1.0] -> int8 (quantize dùng scale/zero-point của model) */
static inline void quantize_f32_to_i8(const void    *handle,
                                       const float   *restrict src,
                                       int            count,
                                       int8_t        *restrict dst)
{
    mtb_ml_model_t *model = *(mtb_ml_model_t **)handle;
    float in_scale  = model->input_scale;
    int   in_offset = model->input_zero_point;

    for (int i = 0; i < count; i++) {
        float v = (src[i] / in_scale) + in_offset;
        if      (v >  127.0f) dst[i] =  127;
        else if (v < -128.0f) dst[i] = -128;
        else                  dst[i] = (int8_t)v;
    }
}

/* Dequantize int8 -> float */
static inline void dequantize_d8_to_f32(const int8_t *restrict src,
                                          float        *restrict dst,
                                          int count, float scale, int offset)
{
    while (count-- > 0)
        *dst++ = (*src++ - offset) * scale;
}


// Candidate detection data structure
typedef struct {
    int index;
    float score;
    int class_id;
    float box[4];   // [x, y, w, h]
} candidate_detection_t;

/*
 * YOLOv8 raw layout:
 * input shape: [8, 2100]
 *
 * channel 0: x
 * channel 1: y
 * channel 2: w
 * channel 3: h
 * channel 4: class0
 * channel 5: class1
 * channel 6: class2
 * channel 7: class3
 *
 * Memory layout is channel-major:
 * input[channel * detection_count + detection]
 */
static inline float get_yolov8_value(const float *restrict input,
                                     int detection_count,
                                     int channel,
                                     int detection)
{
    return input[channel * detection_count + detection];
}

static inline void detectionfilter_yolov8_f32(const float *restrict input,
                                              float *restrict output,
                                              int detection_count,
                                              int channel_count,
                                              int max_detections,
                                              float threshold,
                                              float iou_threshold,
                                              int include_detected_flag)
{
    int class_offset = 4;
    int num_classes = channel_count - class_offset;

    int keep_indices[detection_count];
    int num_candidates = 0;

    // 1. Filter by max class confidence
    printf("****1*****\n");
    for (int i = 0; i < detection_count; ++i) {
        float max_conf = 0.0f;
        
        if(i<10){
	        printf("\r\nDET %d\r\n", i);
	
		    printf("x=%f y=%f w=%f h=%f\r\n",
		           get_yolov8_value(input, detection_count, 0, i),
		           get_yolov8_value(input, detection_count, 1, i),
		           get_yolov8_value(input, detection_count, 2, i),
		           get_yolov8_value(input, detection_count, 3, i));
	    }
	    

        for (int j = 0; j < num_classes; ++j) {
            float current_conf = get_yolov8_value(input, detection_count, class_offset + j, i);
            //printf("class %d conf = %f\r\n", j, current_conf);
            if (current_conf > max_conf) {
                max_conf = current_conf;
            }
        }

        // YOLOv8 has no objectness, so score = max class confidence
        if (max_conf >= threshold) {
            keep_indices[num_candidates] = i;
            num_candidates++;
        }
    }
    //printf("num_candidates = %d\r\n", num_candidates);

    int output_confidence_count = channel_count; // x,y,w,h + classes = 8
    int output_count = include_detected_flag ? output_confidence_count + 1 : output_confidence_count;

    if (num_candidates == 0) {
        for (int i = 0; i < max_detections * output_count; ++i) {
            output[i] = 0.0f;
        }
        return;
    }

    int candidate_indices[num_candidates];
    float candidate_scores[num_candidates];
    int candidate_classes[num_candidates];
    float candidate_boxes[num_candidates][4];

    // 2. Store candidates
    printf("****2*****\n");
    for (int idx = 0; idx < num_candidates; ++idx) {
        int i = keep_indices[idx];

        int max_class = -1;
        float max_conf = 0.0f;

        for (int j = 0; j < num_classes; ++j) {
            float current_conf = get_yolov8_value(input, detection_count, class_offset + j, i);
            if (current_conf > max_conf) {
                max_conf = current_conf;
                max_class = j;
            }
        }

        candidate_indices[idx] = i;
        candidate_scores[idx] = max_conf;
        candidate_classes[idx] = max_class;

        for (int j = 0; j < 4; ++j) {
            candidate_boxes[idx][j] = get_yolov8_value(input, detection_count, j, i);
        }
    }

    // 3. Sort candidates by score, descending
    printf("****3*****\n");
    for (int i = 0; i < num_candidates - 1; ++i) {
        int max_idx = i;

        for (int j = i + 1; j < num_candidates; ++j) {
            if (candidate_scores[j] > candidate_scores[max_idx]) {
                max_idx = j;
            }
        }

        if (max_idx != i) {
            int temp_idx = candidate_indices[i];
            candidate_indices[i] = candidate_indices[max_idx];
            candidate_indices[max_idx] = temp_idx;

            float temp_score = candidate_scores[i];
            candidate_scores[i] = candidate_scores[max_idx];
            candidate_scores[max_idx] = temp_score;

            int temp_class = candidate_classes[i];
            candidate_classes[i] = candidate_classes[max_idx];
            candidate_classes[max_idx] = temp_class;

            float temp_box[4];
            for (int k = 0; k < 4; ++k) {
                temp_box[k] = candidate_boxes[i][k];
                candidate_boxes[i][k] = candidate_boxes[max_idx][k];
                candidate_boxes[max_idx][k] = temp_box[k];
            }
        }
    }

    // 4. NMS, compare only same class
    printf("****4*****\n");
    int suppressed[num_candidates];

    for (int i = 0; i < num_candidates; ++i) {
        suppressed[i] = 0;
    }

    for (int i = 0; i < num_candidates; ++i) {
        if (suppressed[i]) {
            continue;
        }

        float *box_i = candidate_boxes[i];
        int class_i = candidate_classes[i];

        float ix = box_i[0];
        float iy = box_i[1];
        float iw = box_i[2];
        float ih = box_i[3];

        float x1_i = ix - iw / 2.0f;
        float y1_i = iy - ih / 2.0f;
        float x2_i = ix + iw / 2.0f;
        float y2_i = iy + ih / 2.0f;

        for (int j = i + 1; j < num_candidates; ++j) {
            if (suppressed[j] || candidate_classes[j] != class_i) {
                continue;
            }

            float *box_j = candidate_boxes[j];

            float jx = box_j[0];
            float jy = box_j[1];
            float jw = box_j[2];
            float jh = box_j[3];

            float x1_j = jx - jw / 2.0f;
            float y1_j = jy - jh / 2.0f;
            float x2_j = jx + jw / 2.0f;
            float y2_j = jy + jh / 2.0f;

            float x1 = fmaxf(x1_i, x1_j);
            float y1 = fmaxf(y1_i, y1_j);
            float x2 = fminf(x2_i, x2_j);
            float y2 = fminf(y2_i, y2_j);

            float inter_area = (x2 > x1 && y2 > y1) ? (x2 - x1) * (y2 - y1) : 0.0f;

            float area_i = iw * ih;
            float area_j = jw * jh;
            float union_area = area_i + area_j - inter_area;

            float iou = (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;

            if (iou > iou_threshold) {
                suppressed[j] = 1;
            }
        }
    }

    // 5. Write output as [x,y,w,h,class0,class1,class2,class3,detect_flag] x max_detections
    printf("****5*****\n");
    int output_idx = 0;

    for (int i = 0; i < num_candidates && output_idx < max_detections; ++i) {
        if (!suppressed[i]) {
            int input_idx = candidate_indices[i];

            // Copy x,y,w,h
            for (int j = 0; j < 4; ++j) {
                output[j * max_detections + output_idx] =
                    get_yolov8_value(input, detection_count, j, input_idx);
            }

            // Copy class scores directly, no objectness multiplication
            for (int j = 0; j < num_classes; ++j) {
                output[(class_offset + j) * max_detections + output_idx] =
                    get_yolov8_value(input, detection_count, class_offset + j, input_idx);
            }

            // Add detect flag
            if (include_detected_flag) {
                output[channel_count * max_detections + output_idx] = 1.0f;
            }

            output_idx++;
        }
    }

    // 6. Fill remaining output slots with zero
    printf("****6*****\n");
    for (; output_idx < max_detections; ++output_idx) {
        for (int j = 0; j < output_count; ++j) {
            output[j * max_detections + output_idx] = 0.0f;
        }
    }
}

/* Run model và copy raw output vào dst */
static inline void mtb_model_raw(const void *handle,
                                  const void *restrict src,
                                  int         src_count,
                                  void       *restrict dst,
                                  int         dst_byte_count)
{
    mtb_ml_model_t *model = *(mtb_ml_model_t **)handle;
    mtb_ml_model_run(model, (MTB_ML_DATA_T *)src);
    memcpy(dst, model->output, dst_byte_count);
}

/* ====================================================================
 * mtb lifecycle helpers (giữ nguyên từ YOLOv5n)
 * ==================================================================== */
static inline void mtb_model_free(const void *handle)
{
    mtb_ml_model_t *model = *(mtb_ml_model_t **)handle;
    mtb_ml_model_deinit(model);
    mtb_ml_deinit();
    if (IMAI_mtb_models_count > 0)
        IMAI_mtb_models_count--;
}

int32_t IMAI_mtb_models_count = 0;
mtb_ml_model_t *IMAI_mtb_models[IMAI_MAX_MTB_MODELS];

int mtb_init(const void *handle,
             uint8_t *model_bin, unsigned int model_size,
             uint8_t *arena_buffer, int arena_size,
             int npu_priority, char model_name[])
{
    printf("/******************** mtb_init() ********************/\n");
    printf("handle       = %p\n", handle);
    printf("model_bin    = %p\n", model_bin);
    printf("arena_buffer = %p\n", arena_buffer);
    printf("arena_size   = %d\n", arena_size);

    mtb_ml_model_t **model_obj = (mtb_ml_model_t **)handle;

    mtb_ml_model_bin_t model_bin_t = {
        .model_bin  = model_bin,
        .model_size = model_size,
        .arena_size = arena_size
    };
    strncpy(model_bin_t.name, model_name, MTB_ML_MODEL_NAME_LEN - 1);
    model_bin_t.name[MTB_ML_MODEL_NAME_LEN - 1] = '\0';

    mtb_ml_model_buffer_t buffer = {
        .tensor_arena      = arena_buffer,
        .tensor_arena_size = arena_size
    };

    printf("Before mtb_ml_model_init\n");
    cy_rslt_t r1 = mtb_ml_model_init(&model_bin_t, &buffer, model_obj);
    printf("After  mtb_ml_model_init: 0x%08lx\n", (unsigned long)r1);
    if (r1 != CY_RSLT_SUCCESS) return IPWIN_RET_ERROR;

    printf("Before mtb_ml_init\n");
    cy_rslt_t r2 = mtb_ml_init(npu_priority);
    printf("After  mtb_ml_init: 0x%08lx\n", (unsigned long)r2);
    if (r2 != CY_RSLT_SUCCESS) return IPWIN_RET_ERROR;

#ifdef IMAI_PROFILING
    if (mtb_ml_model_profile_config(*model_obj, MTB_ML_PROFILE_ENABLE_MODEL) != CY_RSLT_SUCCESS)
        return IPWIN_RET_ERROR;
    IMAI_mtb_models[IMAI_mtb_models_count++] = *model_obj;
#endif

    printf("/******************** mtb_init() done ***************/\n");
    return 0;
}

/* ====================================================================
 * Public API
 * ==================================================================== */

#ifndef __CLOSE_HOOKS
    #define __CLOSE_HOOKS() do { } while(0)
#endif
#define __RETURN_ERROR(_exp) \
    do { int __r = (_exp); if (__r < 0) { __CLOSE_HOOKS(); return __r; } } while(0)

/*
 * IMAI_compute
 *
 *   [IMAI_INPUT_UINT8]  datain : const uint8_t* — ảnh raw [320,320,3]
 *
 *   dataout : float[8, 2100]  — raw dequantized output
 *             layout: [channel][detection]
 *             channel 0-3 : x, y, w, h  (normalized 0..1)
 *             channel 4-7 : ThanLa, RaNu, NoHoa, Harvest (class scores)
 *
 *   Không có NMS — consumer code tự lọc theo threshold và IoU.
 */

void IMAI_compute(const uint8_t *restrict datain, float *restrict dataout)
{
    /* 1. uint8 -> int8 (range shift) */
    cast_u8_to_i8(datain, _K1, 307200);
    //cast_u8_to_i8_yolov8(_K6, datain, _K1, 307200);
    //mtb_model_uint8_to_quantized_int8_like_yolov8(_K6, datain, 307200, _K5, 16800, _K1);

    /* 2. Inference -> raw int8 output vào _K5 */
    mtb_model_raw(_K6, _K1, 307200, _K5, YOLOv8_RAW_BYTES);
    for (int i = 0; i < 20; i++) {
	    printf("_K5[%d] = %d\r\n", i, _K5[i]);
	}

    /* 3. Dequantize int8 -> float, ghi vào dataout */
    dequantize_d8_to_f32(_K5, _K7, YOLOv8_RAW_BYTES, YOLOv8_DEQUANT_SCALE, YOLOv8_DEQUANT_OFFSET);
    printf("dequantize ok\n");
    for (int i = 0; i < 20; i++) {
	    printf("_K7[%d] = %f\r\n", i, _K7[i]);
	}

    detectionfilter_yolov8_f32(_K7, dataout, 2100, 8, 5, 0.3f, 0.5f, true);
    printf("detection filter ok\n");
}


void IMAI_finalize(void)
{
    mtb_model_free(_K6);
}

__attribute__((noinline)) int IMAI_init(void)
{
    printf("/****************** IMAI_init() *******************/\n");
    printf("_buffer   = %p\n", _buffer);
    printf("_state    = %p\n", _state);
    printf("_K3_flash = %p\n", _K3_flash);

    __RETURN_ERROR(mtb_init(_K6,
                            _K3,  READONLY_SIZE,
                            _K2,  STATE_SIZE - 16,  // trừ 16 bytes cho model handle
                            3,
                            "yolov8n-size320-4cls_int8_int8x8"));
    return 0;
}

static IMAI_api_def _IMAI_api_def = {
    .api_ver = 1,
    .id = {0xe5, 0x9a, 0x59, 0x0f, 0x7f, 0x1c, 0xa4, 0x47, 0x97, 0x28, 0xf8, 0xa6, 0xfb, 0x50, 0xfe, 0x58},
    .api_type = IMAI_API_TYPE_FUNCTION,
    .prefix = "IMAI_",
    .buffer_mem = {
        .size = BUFFER_SIZE,
        .peak_usage = BUFFER_SIZE,
    },
    .static_mem = {
        .size = STATE_SIZE,
        .peak_usage = STATE_SIZE,
    },
    .readonly_mem = {
        .size = READONLY_SIZE,
        .peak_usage = READONLY_SIZE,
    },
    .func_count = 3,
    .func_list = (IMAI_func_def[]) {
        {
            .name = "IMAI_compute",
            .description = "",
            .fn_ptr = IMAI_compute,
            .attrib = 2,
            .param_count = 2,
            .param_list = (IMAI_param_def[]) {
                {
                    .name = "datain",
                    .attrib = IMAI_PARAM_INPUT,
                    .rank = 3,
                    .shape = (IMAI_shape_dim[]) {
                        {
                            .name = "",
                            .size = 3,
                        },
                        {
                            .name = "",
                            .size = 320,
                        },
                        {
                            .name = "",
                            .size = 320,
                        },
                    },
                    .count = BUFFER_SIZE,
                    .type_id = IMAGINET_TYPES_UINT8,
                    .frequency = 15,
                    .shift = 0,
                    .scale = 1,
                    .offset = 0,
                },
                {
                    .name = "dataout",
                    .attrib = IMAI_PARAM_OUTPUT,
                    .rank = 2,
                    .shape = (IMAI_shape_dim[]) {
                        {
                            .name = "",
                            .size = 5,
                        },
                        {
                            .name = "",
                            .size = 9,
                            .labels = (label_text_t[]) { "x","y","width","height","ThanLa","RaNu","NoHoa","Harvest","detect_flat"},
                        },
                    },
                    .count = 45,
                    .type_id = IMAGINET_TYPES_FLOAT32,
                    .frequency = 15,
                    .shift = 0,
                    .scale = 1,
                    .offset = 0,
                },
            },
        },
        {
            .name = "IMAI_finalize",
            .description = "Closes and flushes streams, free any heap allocated memory.",
            .fn_ptr = IMAI_finalize,
            .attrib = 10,
            .param_count = 0,
            .param_list = (IMAI_param_def[]) {
            },
        },
        {
            .name = "IMAI_init",
            .description = "Initializes buffers to initial state.",
            .fn_ptr = IMAI_init,
            .attrib = 7,
            .param_count = 0,
            .param_list = (IMAI_param_def[]) {
            },
        },
    },
};

IMAI_api_def *IMAI_api(void) {
    return &_IMAI_api_def;
}
