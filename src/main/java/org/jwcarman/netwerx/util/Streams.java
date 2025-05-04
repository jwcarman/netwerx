package org.jwcarman.netwerx.util;

import java.util.Iterator;
import java.util.Spliterators;
import java.util.function.BiFunction;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class Streams {

// -------------------------- STATIC METHODS --------------------------

    public static <A,B> Stream<Pair<A,B>> zip(Stream<? extends A> a,
                                                Stream<? extends B> b) {
        return zip(a, b, Pair::new);
    }

    public static <A, B, C> Stream<C> zip(
            Stream<? extends A> a,
            Stream<? extends B> b,
            BiFunction<? super A, ? super B, ? extends C> zipper) {
        Iterator<? extends A> iterA = a.iterator();
        Iterator<? extends B> iterB = b.iterator();

        return StreamSupport.stream(
                Spliterators.spliteratorUnknownSize(new Iterator<>() {

                    @Override
                    public boolean hasNext() {
                        return iterA.hasNext() && iterB.hasNext();
                    }

                    @Override
                    public C next() {
                        return zipper.apply(iterA.next(), iterB.next());
                    }

                }, 0), false);
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Streams() {
        // Prevent instantiation
    }

// -------------------------- INNER CLASSES --------------------------

    public record Pair<A,B>(A left, B right) {}

}
