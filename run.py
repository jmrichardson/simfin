### Initial run
from simfin import *
SimFin().extract().flatten().outliers().missing_rows().history().csv()

### Example runs after initial invocation
# SimFin().flatten().query(["MSFT"]).csv()
# SimFin().flatten().sample().csv()
