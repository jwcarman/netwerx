package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LayerUpdate<M extends Matrix<M>> {

// -------------------------- OTHER METHODS --------------------------

    private final Map<String,M> gradients = new HashMap<>();


    public void addGradient(String name, M gradient) {
        gradients.put(name, gradient);
    }

    public M gradient(String name) {
        if (!gradients.containsKey(name)) {
            throw new IllegalArgumentException("No gradient found for name: " + name);
        }
        return gradients.get(name);
    }

    public List<String> gradientNames() {
        return List.copyOf(gradients.keySet());
    }

}
