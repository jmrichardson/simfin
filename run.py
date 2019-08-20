from simfin import *
SimFin().extract().flatten().outliers().missing_rows().history().csv()
# SimFin().extract().flatten().outliers().history().impute().csv()
# SimFin().extract().flatten().outliers().history().csv()

### Example runs after initial invocation
# SimFin().flatten().query(["MSFT"]).csv()
# SimFin().flatten().sample().csv()
#
## Keeping state
# simfin = SimFin().flatten()
# simfin = simfin.query(["FLEX"])
# simfin = simfin.missing_rows().history()
# simfin = simfin.csv()

