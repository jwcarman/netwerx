package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;
import org.jwcarman.netwerx.regression.RegressionModelConfig;

class DefaultRegressionModelConfig implements RegressionModelConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.mse();

// --------------------- GETTER / SETTER METHODS ---------------------

    public Loss getLoss() {
        return loss;
    }

// ------------------------ INTERFACE METHODS ------------------------

// --------------------- Interface RegressionModelConfig ---------------------

    @Override
    public DefaultRegressionModelConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

}
