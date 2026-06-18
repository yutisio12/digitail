package com.meterapi.exception;

import lombok.Getter;

@Getter
public class ApiKeyException extends RuntimeException {
    private final String code;
    private final int httpStatus;

    public ApiKeyException(String code, int httpStatus) {
        super(code);
        this.code = code;
        this.httpStatus = httpStatus;
    }

    public ApiKeyException(String code, String message, int httpStatus) {
        super(message);
        this.code = code;
        this.httpStatus = httpStatus;
    }
}
