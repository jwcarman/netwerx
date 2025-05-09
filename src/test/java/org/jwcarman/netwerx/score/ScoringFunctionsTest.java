package org.jwcarman.netwerx.score;

import org.junit.jupiter.api.Test;
import org.jwcarman.netwerx.EpochOutcome;

import static org.assertj.core.api.Assertions.assertThat;

class ScoringFunctionsTest {
    @Test
    void testValidationLoss() {
        var function = ScoringFunctions.validationLoss();
        var outcome = new EpochOutcome(100, 0.5, 0.1, 0.05, 0.02);
        var score = function.score(outcome);
        assertThat(score).isEqualTo(-outcome.validationLoss());
    }

    @Test
    void testTrainingLoss() {
        var function = ScoringFunctions.trainingLoss();
        var outcome = new EpochOutcome(100, 0.5, 0.1, 0.05, 0.02);
        var score = function.score(outcome);
        assertThat(score).isEqualTo(-outcome.trainingLoss());
    }

    @Test
    void testValidationLossWithPenalty() {
        var function = ScoringFunctions.validationLossWithPenalty();
        var outcome = new EpochOutcome(100, 0.5, 0.1, 0.05, 0.02);
        var score = function.score(outcome);
        assertThat(score).isEqualTo(-(outcome.validationLoss() + outcome.regularizationPenalty()));
    }

}