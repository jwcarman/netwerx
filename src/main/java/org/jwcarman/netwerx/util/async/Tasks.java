package org.jwcarman.netwerx.util.async;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.StructuredTaskScope;
import java.util.function.Supplier;

public class Tasks {

// -------------------------- STATIC METHODS --------------------------

    public static <T> List<T> executeAll(List<Supplier<T>> tasks) {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            var subtasks = tasks.stream().map(t -> scope.fork(t::get)).toList();
            scope.join().throwIfFailed();
            return subtasks.stream().map(StructuredTaskScope.Subtask::get).toList();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new TaskExecutionException("Task execution interrupted", e);
        } catch (ExecutionException e) {
            throw new TaskExecutionException("Task execution failed", e);
        }
    }

// --------------------------- CONSTRUCTORS ---------------------------

    private Tasks() {
        // Prevent instantiation
    }

}
