package com.meterapi.exception;

import lombok.Getter;

@Getter
public class OcrLowConfidenceException extends RuntimeException {
    private final String code;
    private final int httpStatus;

    public OcrLowConfidenceException(String code, int httpStatus) {
        super(code);
        this.code = code;
        this.httpStatus = httpStatus;
    }

    public OcrLowConfidenceException(String code, String message, int httpStatus) {
        super(message);
        this.code = code;
        this.httpStatus = httpStatus;
    }
}
