# to use run tensorboard dev upload --logdir logs/fit_*  on shell in the
# directory wher are the logs e.g. [/nobackup/data/andch552/DATA_FILES/TfRecords/derivative_file]

from packaging import version
import sys
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb


experiment_id = "GmXZCkLaR6Sa1o0xJWiVfg"
experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
df = experiment.get_scalars()

print(df_ex)

sys.exit(1)

# print(df["run"].unique())
# print(df["tag"].unique())

dfw = experiment.get_scalars(pivot=True)
dfw
sys.exit(1)
dfw_validation = dfw[dfw.run.str.endswith("/validation")]
# Get the optimizer value for each row of the validation DataFrame.
optimizer_validation = dfw_validation.run.apply(lambda run: run.split(",")[0])

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=dfw_validation, x="step", y="epoch_accuracy",
             hue=optimizer_validation).set_title("accuracy")
plt.subplot(1, 2, 2)
sns.lineplot(data=dfw_validation, x="step", y="epoch_loss",
             hue=optimizer_validation).set_title("loss")
