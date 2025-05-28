package org.jwcarman.netwerx.layer;

import org.jwcarman.netwerx.matrix.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LayerUpdate<M extends Matrix<M>> {

// ------------------------------ FIELDS ------------------------------

    private final Map<String,M> gradients = new HashMap<>();

// -------------------------- STATIC METHODS --------------------------

    public static <M extends Matrix<M>> LayerUpdate<M> aggregate(List<LayerUpdate<M>> updates) {
        return updates.stream().reduce(new LayerUpdate<>(), (acc, update) -> {
            acc.merge(update);
            return acc;
        });
    }


// -------------------------- OTHER METHODS --------------------------

    public void merge(LayerUpdate<M> other) {
        for (String name : other.gradientNames()) {
            addGradient(name, other.gradient(name));
        }
    }

    public LayerUpdate<M> scaled(double weight) {
        LayerUpdate<M> weightedUpdate = new LayerUpdate<>();
        for (Map.Entry<String, M> entry : gradients.entrySet()) {
            weightedUpdate.addGradient(entry.getKey(), entry.getValue().scale(weight));
        }
        return weightedUpdate;
    }

    public List<String> gradientNames() {
        return List.copyOf(gradients.keySet());
    }

    public void addGradient(String name, M gradient) {
        gradients.merge(name, gradient, Matrix::add);
    }

    public M gradient(String name) {
        if (!gradients.containsKey(name)) {
            throw new IllegalArgumentException("No gradient found for name: " + name);
        }
        return gradients.get(name);
    }

}
