package com.meterapi.dto;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class OcrResult {
    private String reading;
    private Double confidence;

    @JsonProperty("bounding_boxes")
    private List<BoundingBox> boundingBoxes;

    @JsonProperty("model_version")
    private String modelVersion;

    @JsonProperty("processing_time_ms")
    private Double processingTimeMs;

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class BoundingBox {
        private List<List<Double>> points;
        private String text;
        private Double confidence;
    }
}
