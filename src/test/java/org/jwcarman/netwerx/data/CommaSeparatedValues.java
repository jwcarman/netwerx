package org.jwcarman.netwerx.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.InputStreamReader;
import java.util.List;
import java.util.function.Function;

public class CommaSeparatedValues {

// -------------------------- STATIC METHODS --------------------------

    /**
     * Loads a CSV resource and maps each record to an object using the provided mapper function.
     *
     * @param resourceName The name of the resource to load.
     * @param mapper       A function that maps a CSVRecord to an object of type T.
     * @param <T>          The type of objects to return.
     * @return A list of objects of type T mapped from the CSV records.
     */
    public static <T> List<T> load(String resourceName, Function<CSVRecord, T> mapper) {
        try (var in = CommaSeparatedValues.class.getResourceAsStream(resourceName);
             var reader = new InputStreamReader(in);
        ) {
            return CSVFormat.DEFAULT.builder()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .get()
                    .parse(reader).stream()
                    .map(mapper)
                    .toList();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load CSV data from " + resourceName, e);
        }
    }

}
