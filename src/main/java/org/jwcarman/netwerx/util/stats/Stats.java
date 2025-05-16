package org.jwcarman.netwerx.util.stats;

import java.util.stream.DoubleStream;

public class Stats {

// ------------------------------ FIELDS ------------------------------

    private long count = 0;
    private double sum = 0.0;
    private double sumSq = 0.0;
    private double min = Double.POSITIVE_INFINITY;
    private double max = Double.NEGATIVE_INFINITY;

// -------------------------- OTHER METHODS --------------------------

    public void accumulate(double value) {
        count++;
        sum += value;
        sumSq += value * value;
        if (value < min) min = value;
        if (value > max) max = value;
    }

    public void combine(Stats other) {
        count += other.count;
        sum += other.sum;
        sumSq += other.sumSq;
        min = Math.min(min, other.min);
        max = Math.max(max, other.max);
    }


    public double maxAbs() {
        return Math.max(Math.abs(min), Math.abs(max));
    }

    public double l2() {
        return Math.sqrt(sumSq);
    }

    public long count() {
        return count;
    }

    public double max() {
        return max;
    }

    public double min() {
        return min;
    }

    public double stddev() {
        return Math.sqrt(variance());
    }

    public double variance() {
        double mean = mean();
        return (sumSq / count) - (mean * mean);
    }

    public double mean() {
        return sum / count;
    }


    public static Stats of(DoubleStream values) {
        Stats stats = values.collect(Stats::new, Stats::accumulate, Stats::combine);
        if(stats.count == 0) {
            throw new IllegalArgumentException("Stats must have at least one value");
        }
        return stats;
    }
}
