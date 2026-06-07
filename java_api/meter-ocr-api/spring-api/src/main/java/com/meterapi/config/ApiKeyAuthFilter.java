package com.meterapi.config;

import com.meterapi.exception.ApiKeyException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;
import java.util.List;

@Component
public class ApiKeyAuthFilter extends OncePerRequestFilter {

    private static final Logger log = LoggerFactory.getLogger(ApiKeyAuthFilter.class);

    private final List<String> validKeys;

    public ApiKeyAuthFilter(@Value("${api.key.valid-keys}") List<String> validKeys) {
        this.validKeys = validKeys;
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain filterChain) throws ServletException, IOException {

        String apiKey = request.getHeader("X-API-KEY");

        if (apiKey == null || apiKey.isBlank()) {
            throw new ApiKeyException("AUTH_MISSING_KEY", "Missing X-API-KEY header", 401);
        }

        if (!validKeys.contains(apiKey)) {
            String masked = apiKey.length() > 4
                    ? apiKey.substring(0, 2) + "****" + apiKey.substring(apiKey.length() - 2)
                    : "****";
            log.warn("Invalid API key attempt: {}", masked);
            throw new ApiKeyException("AUTH_INVALID_KEY", "Invalid API key", 401);
        }

        SecurityContextHolder.getContext().setAuthentication(
                new UsernamePasswordAuthenticationToken(apiKey, null, List.of())
        );

        filterChain.doFilter(request, response);
    }
}
