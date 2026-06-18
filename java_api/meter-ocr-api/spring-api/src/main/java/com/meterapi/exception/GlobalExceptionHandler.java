package com.meterapi.exception;

import com.meterapi.dto.ErrorResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.time.Instant;
import java.util.UUID;

@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    @ExceptionHandler(ApiKeyException.class)
    public ResponseEntity<ErrorResponse> handleApiKey(ApiKeyException ex) {
        return buildError(ex.getCode(), ex.getMessage(), ex.getHttpStatus());
    }

    @ExceptionHandler(OcrServiceException.class)
    public ResponseEntity<ErrorResponse> handleOcrService(OcrServiceException ex) {
        return buildError(ex.getCode(), ex.getMessage(), ex.getHttpStatus());
    }

    @ExceptionHandler(OcrLowConfidenceException.class)
    public ResponseEntity<ErrorResponse> handleLowConfidence(OcrLowConfidenceException ex) {
        return buildError(ex.getCode(), ex.getMessage(), ex.getHttpStatus());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ErrorResponse> handleMaxUpload(MaxUploadSizeExceededException ex) {
        return buildError("FILE_TOO_LARGE", "File size exceeds maximum allowed size", 413);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArg(IllegalArgumentException ex) {
        return buildError("INVALID_REQUEST", ex.getMessage(), 400);
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGeneral(Exception ex) {
        log.error("Unhandled exception", ex);
        return buildError("INTERNAL_ERROR", "Internal server error", 500);
    }

    private ResponseEntity<ErrorResponse> buildError(String code, String message, int httpStatus) {
        ErrorResponse.ErrorDetail detail = new ErrorResponse.ErrorDetail(code, message, httpStatus);
        ErrorResponse res = new ErrorResponse(false, UUID.randomUUID().toString(), Instant.now().toString(), detail);
        return ResponseEntity.status(HttpStatus.valueOf(httpStatus)).body(res);
    }
}
