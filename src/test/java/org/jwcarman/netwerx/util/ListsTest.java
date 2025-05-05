package org.jwcarman.netwerx.util;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ListsTest {

    @Test
    void testChunked() {
        var list = List.of(1,2,3,4,5,6,7,8);
        var chunked = Lists.chunked(list, 3);
        assertThat(chunked).hasSize(3);
        assertThat(chunked.get(0)).containsExactly(1, 2, 3);
        assertThat(chunked.get(1)).containsExactly(4, 5, 6);
        assertThat(chunked.get(2)).containsExactly(7, 8);
    }

    @Test
    void testChunkedWithChunkSizeLargerThanList() {
        var list = List.of(1,2,3,4,5,6,7,8);
        var chunked = Lists.chunked(list, 10);
        assertThat(chunked).hasSize(1);
        assertThat(chunked.getFirst()).isSameAs(list);
    }

    @Test
    void testChunkedWithNegativeChunkSize() {
        var list = List.of(1,2,3,4,5,6,7,8);
        assertThatThrownBy(() -> Lists.chunked(list, -1))
                .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void testChunkedWithNullList() {
        assertThat(Lists.chunked(null, 3)).isEmpty();
    }

    @Test
    void testChunkedWithEmptyList() {
        var emptyList = List.<Integer>of();
        var chunked = Lists.chunked(emptyList, 3);
        assertThat(chunked).isEmpty();
    }

}