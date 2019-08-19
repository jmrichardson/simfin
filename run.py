from simfin import *
simfin = SimFin().extract().flatten().outliers().missing_rows().history().csv()
