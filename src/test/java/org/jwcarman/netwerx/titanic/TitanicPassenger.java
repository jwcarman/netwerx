package org.jwcarman.netwerx.titanic;

record TitanicPassenger(String name, int ticketClass, double age, int sex, double fare, int siblingsAndSpouses,
                        int parentsAndChildren, boolean survived) {

}
