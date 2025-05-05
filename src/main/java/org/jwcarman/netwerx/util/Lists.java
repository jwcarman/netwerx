package org.jwcarman.netwerx.util;

import java.util.List;

public class Lists {
    private Lists() {
        // Prevent instantiation
    }

    public static <T> List<List<T>> chunked(List<T> original, int chunkSize) {
        if (chunkSize <= 0) {
            throw new IllegalArgumentException("Chunk size must be greater than zero.");
        }
        if (original == null || original.isEmpty()) {
            return List.of();
        }
        if(chunkSize >= original.size()) {
            return List.of(original);
        }

        int totalChunks = (int) Math.ceil((double) original.size() / chunkSize);
        List<List<T>> chunks = new java.util.ArrayList<>(totalChunks);

        for (int i = 0; i < totalChunks; i++) {
            int start = i * chunkSize;
            int end = Math.min(start + chunkSize, original.size());
            chunks.add(original.subList(start, end));
        }

        return chunks;
    }
}
