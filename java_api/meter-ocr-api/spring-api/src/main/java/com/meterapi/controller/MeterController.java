package com.meterapi.controller;

import com.meterapi.dto.MeterReadResponse;
import com.meterapi.service.MeterService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class MeterController {

    private final MeterService meterService;

    public MeterController(MeterService meterService) {
        this.meterService = meterService;
    }

    @PostMapping("/meter/read")
    public ResponseEntity<MeterReadResponse> readMeter(
            @RequestPart("image") MultipartFile image,
            @RequestParam(name = "meter_type", defaultValue = "electric") String meterType) {

        MeterReadResponse response = meterService.processMeterRead(image, meterType);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of("status", "ok", "service", "meter-ocr-api"));
    }
}
