package org.jwcarman.netwerx.regression;

/**
 * Regression model evaluation statistics.
 * <p>
 * Captures typical regression metrics: mean squared error (MSE), mean absolute error (MAE), and R² (coefficient of determination).
 *
 * @param mse Mean Squared Error
 * @param mae Mean Absolute Error
 * @param r2  R² (coefficient of determination, closer to 1 is better)
 */
public record RegressionModelStats(double mse, double mae, double r2) {

    public static RegressionModelStats of(double[] predictions, double[] targets) {
        final int n = predictions.length;
        double sumSquaredErrors = 0.0;
        double sumAbsoluteErrors = 0.0;
        double sumTargets = 0.0;

        for (int i = 0; i < n; i++) {
            double y = targets[i];
            double yHat = predictions[i];
            double error = yHat - y;
            sumSquaredErrors += error * error;
            sumAbsoluteErrors += Math.abs(error);
            sumTargets += y;
        }

        double mse = sumSquaredErrors / n;
        double mae = sumAbsoluteErrors / n;

        double meanTarget = sumTargets / n;
        double totalVariance = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = targets[i] - meanTarget;
            totalVariance += diff * diff;
        }

        double r2 = totalVariance == 0.0 ? 0.0 : 1.0 - (sumSquaredErrors / totalVariance);

        return new RegressionModelStats(mse, mae, r2);
    }

}