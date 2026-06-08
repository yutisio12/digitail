package com.meterapi.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class ErrorResponse {
    private boolean success;

    @JsonProperty("request_id")
    private String requestId;

    private String timestamp;

    private ErrorDetail error;

    @Data
    @AllArgsConstructor
    public static class ErrorDetail {
        private String code;
        private String message;

        @JsonProperty("http_status")
        private int httpStatus;
    }
}
