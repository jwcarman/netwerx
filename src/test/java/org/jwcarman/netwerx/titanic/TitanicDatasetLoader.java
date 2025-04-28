package org.jwcarman.netwerx.titanic;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class TitanicDatasetLoader {

    private static final String TRAINING_FILE_PATH = "/dataset/titanic/train.csv";
    private static final String TEST_FILE_PATH = "/dataset/titanic/test.csv";
    public static final String AGE = "Age";
    public static final String FARE = "Fare";
    public static final double DEFAULT_AGE = 30.0;
    public static final String PARENTS_AND_CHILDREN = "Parch";
    public static final String SIBLINGS_AND_SPOUSES = "SibSp";
    public static final String TICKET_CLASS = "Pclass";
    public static final String MALE = "male";
    public static final String SEX = "Sex";
    public static final String SURVIVED = "Survived";
    public static final String TRUE = "1";
    public static final String NAME = "Name";

    public static List<TitanicPassenger> loadTrainingPassengers() throws IOException {
        return loadPassengers(TRAINING_FILE_PATH);
    }

    public static List<TitanicPassenger> loadTestPassengers() throws IOException {
        return loadPassengers(TEST_FILE_PATH);
    }

    private static List<TitanicPassenger> loadPassengers(String resourceName) throws IOException {

        try (Reader reader = new InputStreamReader(TitanicDatasetLoader.class.getResourceAsStream(resourceName));
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            final List<TitanicPassenger> passengers = new ArrayList<>();

            for (CSVRecord csv : csvParser) {
                var name = csv.get(NAME);
                var fare = csv.get(FARE).isEmpty() ? 32.0 : Double.parseDouble(csv.get(FARE));
                var ticketClass = Integer.parseInt(csv.get(TICKET_CLASS));
                var age = csv.get(AGE).isEmpty() ? DEFAULT_AGE : Double.parseDouble(csv.get(AGE));
                var parentsAndChildren = csv.get(PARENTS_AND_CHILDREN).isEmpty() ? 0 : Integer.parseInt(csv.get(PARENTS_AND_CHILDREN));
                var siblingsAndSpouses = csv.get(SIBLINGS_AND_SPOUSES).isEmpty() ? 0 : Integer.parseInt(csv.get(SIBLINGS_AND_SPOUSES));
                var sex = MALE.equalsIgnoreCase(csv.get(SEX)) ? 1 : 0;
                var survived = csv.isSet(SURVIVED) && TRUE.equals(csv.get(SURVIVED));
                passengers.add(new TitanicPassenger(name, ticketClass, age, sex, fare, parentsAndChildren, siblingsAndSpouses, survived));
            }
            return passengers;
        }
    }
}