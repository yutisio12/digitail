package com.meterapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;

import java.util.List;

@Data
@AllArgsConstructor
public class MeterReadResponse {
    private boolean success;

    @JsonProperty("request_id")
    private String requestId;

    private String timestamp;

    private MeterData data;

    @JsonProperty("ocr_meta")
    private OcrMeta ocrMeta;

    @Data
    @AllArgsConstructor
    public static class MeterData {
        @JsonProperty("meter_reading")
        private String meterReading;

        private String unit;

        @JsonProperty("meter_type")
        private String meterType;

        private Double confidence;

        @JsonProperty("bounding_boxes")
        private List<BoundingBoxDTO> boundingBoxes;

        @Data
        @AllArgsConstructor
        public static class BoundingBoxDTO {
            private List<List<Double>> points;
            private String text;
            private Double confidence;
        }
    }

    @Data
    @AllArgsConstructor
    public static class OcrMeta {
        @JsonProperty("model_version")
        private String modelVersion;

        @JsonProperty("processing_time_ms")
        private Double processingTimeMs;
    }
}
