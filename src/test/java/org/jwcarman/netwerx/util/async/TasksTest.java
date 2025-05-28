package org.jwcarman.netwerx.util.async;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.function.Supplier;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.awaitility.Awaitility.await;

class TasksTest {

    @Test
    void shoulExecuteAllTasks() {
        List<Supplier<String>> tasks = List.of(
                () -> "Task 1 completed",
                () -> "Task 2 completed",
                () -> "Task 3 completed"
        );

        var results = Tasks.executeAll(tasks);
        assertThat(results).containsExactly("Task 1 completed", "Task 2 completed", "Task 3 completed");
    }

    @Test
    void shouldThrowTaskExecutionExceptionIfTaskFails() {
        List<Supplier<String>> tasks = List.of(
                () -> {
                    throw new IllegalArgumentException("I don't like you!");
                }
        );

        assertThatThrownBy(() -> Tasks.executeAll(tasks))
                .isInstanceOf(TaskExecutionException.class);
    }

    @Test
    void shouldThrowTaskExecutionExceptionIfTaskIsInterrupted() throws Exception {
        var latch = new CountDownLatch(1);


        List<Supplier<String>> tasks = List.of(
                () -> {

                    latch.countDown();
                    await().atMost(Duration.ofSeconds(10));
                    return "This will not complete!";
                }
        );


        var thread = new Thread(() -> assertThatThrownBy(() -> Tasks.executeAll(tasks))
                .isInstanceOf(TaskExecutionException.class));

        thread.start();
        latch.await();
        thread.interrupt();
        thread.join();
    }

}