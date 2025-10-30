# Load-Information-Repair-Method-for-Heavy-Duty-Diesel-Vehicles
You will be able to find my draft paper here
The load information of heavy-duty diesel vehicles is often partially missing during actual
operation due to sensor failures or data transmission issues, severely hindering the accuracy
of vehicle emission assessment and energy consumption management. To address this prob-
lem, this study proposes a data-driven load information repair method integrating working
condition division, Principal Component Analysis (PCA), and K-means clustering. Using
real-world operational data from 100 heavy-duty diesel vehicles as the research object, data
cleaning and preprocessing were first conducted. To address the continuous nature of the
data, an innovative strategy was proposed to divide the data into working condition segments
using prolonged idling (speed < 1 km/h and duration > 15 minutes) as nodes, transform-
ing the continuous data stream into discrete trip segments. By analyzing four load-related
variables and applying PCA for dimensionality reduction, the cumulative variance contri-
bution rate of the first two principal components reached 89.87%, effectively extracting key
features. After systematically comparing six clustering algorithms, K-means was selected,
and the optimal number of clusters was determined to be K=2. Finally, by comparing the
clustering results with real load data, a high coincidence rate was demonstrated (accuracy
reached 95.8%), verifying the accuracy and effectiveness of the proposed method in repairing
missing load information. This provides a reliable technical pathway for vehicle big data
analysis and the imputation of key information.
Index Terms: Heavy-duty diesel vehicles; Load information repair; Working condition
division; Principal Component Analysis (PCA); K-means clustering; Data mining

You can find almost all of my code here.
