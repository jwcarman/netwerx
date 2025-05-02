package org.jwcarman.netwerx.def;

import org.jwcarman.netwerx.classification.multi.MultiClassifierConfig;
import org.jwcarman.netwerx.loss.Loss;
import org.jwcarman.netwerx.loss.Losses;

class DefaultMultiClassifierConfig implements MultiClassifierConfig {

// ------------------------------ FIELDS ------------------------------

    private Loss loss = Losses.cce();
    private int outputClasses = 3;

// --------------------- GETTER / SETTER METHODS ---------------------



    public Loss getLoss() {
        return loss;
    }

    public int getOutputClasses() {
        return outputClasses;
    }


// -------------------------- OTHER METHODS --------------------------

    @Override
    public DefaultMultiClassifierConfig loss(Loss loss) {
        this.loss = loss;
        return this;
    }

    @Override
    public DefaultMultiClassifierConfig outputClasses(int outputClasses) {
        this.outputClasses = outputClasses;
        return this;
    }

}
