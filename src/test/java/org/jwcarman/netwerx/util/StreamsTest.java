package org.jwcarman.netwerx.util;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

class StreamsTest {

    @Test
    void zip_shouldZipUntilShortestStreamEnds() {
        var a = Stream.of("a", "b", "c");
        var b = Stream.of(1, 2);

        var zipped = Streams.zip(a, b, (s, i) -> s + i).toList();

        assertThat(zipped).containsExactly("a1", "b2");
    }

    @Test
    void zip_shouldReturnEmptyWhenBothStreamsAreEmpty() {
        var a = Stream.<String>empty();
        var b = Stream.<Integer>empty();

        var zipped = Streams.zip(a, b, (s, i) -> s + i).toList();

        assertThat(zipped).isEmpty();
    }

    @Test
    void zip_shouldCombineEqualLengthStreams() {
        List<String> result = Streams.zip(
                Stream.of(1, 2, 3),
                Stream.of("a", "b", "c"),
                (i, s) -> i + s
        ).toList();

        assertThat(result).containsExactly("1a", "2b", "3c");
    }

    @Test
    void zip_shouldStopAtShortestStream() {
        List<String> result = Streams.zip(
                Stream.of(1, 2),
                Stream.of("a", "b", "c"),
                (i, s) -> i + s
        ).toList();

        assertThat(result).containsExactly("1a", "2b");
    }

    @Test
    void zip_shouldReturnEmptyWhenOneStreamIsEmpty() {
        List<String> result = Streams.zip(
                Stream.of(),
                Stream.of("a", "b"),
                (i, s) -> i + s
        ).toList();

        assertThat(result).isEmpty();
    }

    @Test
    void zip_shouldCombineElementsFromBothStreams() {
        var streamA = Stream.of("a", "b", "c");
        var streamB = Stream.of(1, 2, 3);

        var result = Streams.zip(streamA, streamB, (s, i) -> s + i).toList();

        assertThat(result).containsExactly("a1", "b2", "c3");
    }

    @Test
    void zip_shouldShortCircuitToShorterStream() {
        var streamA = Stream.of("x", "y");
        var streamB = Stream.of(10, 20, 30);

        var result = Streams.zip(streamA, streamB, (s, i) -> s + i).toList();

        assertThat(result).containsExactly("x10", "y20"); // Only as long as shortest stream
    }

    @Test
    void zip_withEmptyStream_shouldReturnEmptyStream() {
        var streamA = Stream.empty();
        var streamB = Stream.of(1, 2, 3);

        var result = Streams.zip(streamA, streamB, (a, b) -> a.toString() + b).toList();

        assertThat(result).isEmpty();
    }

    @Test
    void zip_canZipDifferentTypes() {
        var streamA = Stream.of("apple", "banana");
        var streamB = Stream.of(3.14, 2.71);

        var result = Streams.zip(streamA, streamB, (a, b) -> a + "@" + b).toList();

        assertThat(result).containsExactly("apple@3.14", "banana@2.71");
    }

    @Test
    void zip_withInfiniteStream_shouldStillShortCircuit() {
        var streamA = Stream.of("one", "two");
        var streamB = IntStream.iterate(1, i -> i + 1).boxed(); // infinite stream

        var result = Streams.zip(streamA, streamB, (s, i) -> s + ":" + i).toList();

        assertThat(result).containsExactly("one:1", "two:2");
    }
}