| experiment | model | mean [95% bootstrap CI] | notes |
|---|---|---|---|
| D: final return (last 3 eps) | sac | -522 [-673, -373] | IQM -495, n=5 |
| D: final return (last 3 eps) | neural_hamiltonian | -382 [-389, -375] | IQM -380, n=5 |
| D: final return (last 3 eps) | legacy_latent_ode | -381 [-388, -375] | IQM -380, n=5 |
| D: final return (last 3 eps) | neural_ode | -382 [-388, -375] | IQM -380, n=5 |
| D: final return (last 3 eps) | mlp_ensemble | -382 [-390, -376] | IQM -381, n=5 |
| D: final return (last 3 eps) | physics_residual | -382 [-390, -375] | IQM -381, n=5 |
| D: final return (last 3 eps) | physics | -382 [-390, -375] | IQM -381, n=5 |
| H1 low-data return @50 | physics | -382 [-387, -377] | solved 5/5 |
| H1 low-data return @100 | physics | -382 [-387, -377] | solved 5/5 |
| H1 low-data return @200 | physics | -381 [-387, -376] | solved 5/5 |
| H1 low-data return @50 | physics_residual | -384 [-392, -374] | solved 5/5 |
| H1 low-data return @100 | physics_residual | -360 [-391, -310] | solved 5/5 |
| H1 low-data return @200 | physics_residual | -382 [-386, -376] | solved 5/5 |
| H1 low-data return @50 | neural_hamiltonian | -361 [-394, -308] | solved 5/5 |
| H1 low-data return @100 | neural_hamiltonian | -381 [-392, -374] | solved 5/5 |
| H1 low-data return @200 | neural_hamiltonian | -381 [-388, -375] | solved 5/5 |
| H1 low-data return @50 | mlp_ensemble | -675 [-1003, -368] | solved 3/5 |
| H1 low-data return @100 | mlp_ensemble | -759 [-1136, -381] | solved 2/5 |
| H1 low-data return @200 | mlp_ensemble | -381 [-387, -375] | solved 5/5 |
| H1 low-data return @50 | neural_ode | -449 [-544, -388] | solved 4/5 |
| H1 low-data return @100 | neural_ode | -703 [-1068, -406] | solved 2/5 |
| H1 low-data return @200 | neural_ode | -360 [-384, -318] | solved 5/5 |
| H1 low-data return @50 | legacy_latent_ode | -995 [-1353, -637] | solved 1/5 |
| H1 low-data return @100 | legacy_latent_ode | -1405 [-1577, -1093] | solved 0/5 |
| H1 low-data return @200 | legacy_latent_ode | -406 [-538, -314] | solved 4/5 |
| C: H=100 error, 400 transitions | physics | 2.93e-03 [2.66e-05, 6.49e-03] | n=5 |
| C: H=100 error, 5000 transitions | physics | 3.05e-03 [4.91e-05, 6.99e-03] | n=5 |
| C: H=100 error, 400 transitions | physics_residual | 1.70e-02 [4.52e-03, 3.56e-02] | n=5 |
| C: H=100 error, 5000 transitions | physics_residual | 9.18e-03 [2.04e-03, 1.63e-02] | n=5 |
| C: H=100 error, 400 transitions | neural_hamiltonian | 1.58e-03 [6.30e-04, 3.13e-03] | n=5 |
| C: H=100 error, 5000 transitions | neural_hamiltonian | 6.38e-04 [4.75e-04, 7.62e-04] | n=5 |
| C: H=100 error, 400 transitions | mlp_ensemble | 1.32e-02 [7.68e-03, 2.10e-02] | n=5 |
| C: H=100 error, 5000 transitions | mlp_ensemble | 5.04e-03 [4.02e-03, 6.06e-03] | n=5 |
| C: H=100 error, 400 transitions | neural_ode | 4.06e-02 [2.65e-02, 5.38e-02] | n=5 |
| C: H=100 error, 5000 transitions | neural_ode | 2.18e-02 [1.50e-02, 2.86e-02] | n=5 |
| C: H=100 error, 400 transitions | legacy_latent_ode | 1.52e-01 [1.36e-01, 1.80e-01] | n=5 |
| C: H=100 error, 5000 transitions | legacy_latent_ode | 5.79e-02 [4.99e-02, 6.48e-02] | n=5 |
| E: return g x0.5 | physics | adapt -206 [-226, -181]; no-adapt -517 [-659, -374] | n=5 |
| E: return g x0.5 | physics_residual | adapt -206 [-226, -181]; no-adapt -490 [-741, -284] | n=5 |
| E: return g x0.5 | neural_hamiltonian | adapt -206 [-226, -181]; no-adapt -520 [-707, -337] | n=5 |
| E: return g x0.5 | mlp_ensemble | adapt -236 [-260, -212]; no-adapt -376 [-508, -283] | n=5 |
| E: return g x0.5 | neural_ode | adapt -229 [-248, -209]; no-adapt -391 [-486, -298] | n=5 |
| E: return g x0.5 | legacy_latent_ode | adapt -244 [-289, -211]; no-adapt -506 [-596, -389] | n=5 |
| E: return g x1.5 | physics | adapt -443 [-461, -431]; no-adapt -1260 [-1273, -1245] | n=5 |
| E: return g x1.5 | physics_residual | adapt -443 [-461, -432]; no-adapt -1266 [-1279, -1251] | n=5 |
| E: return g x1.5 | neural_hamiltonian | adapt -453 [-470, -437]; no-adapt -1269 [-1273, -1265] | n=5 |
| E: return g x1.5 | mlp_ensemble | adapt -456 [-474, -438]; no-adapt -1274 [-1286, -1266] | n=5 |
| E: return g x1.5 | neural_ode | adapt -490 [-520, -463]; no-adapt -1266 [-1277, -1255] | n=5 |
| E: return g x1.5 | legacy_latent_ode | adapt -477 [-508, -448]; no-adapt -1267 [-1276, -1256] | n=5 |
| E: return g x2.0 | physics | adapt -568 [-578, -557]; no-adapt -1507 [-1512, -1503] | n=5 |
| E: return g x2.0 | physics_residual | adapt -567 [-575, -556]; no-adapt -1502 [-1514, -1492] | n=5 |
| E: return g x2.0 | neural_hamiltonian | adapt -607 [-694, -556]; no-adapt -1502 [-1509, -1496] | n=5 |
| E: return g x2.0 | mlp_ensemble | adapt -682 [-881, -564]; no-adapt -1502 [-1509, -1495] | n=5 |
| E: return g x2.0 | neural_ode | adapt -681 [-762, -599]; no-adapt -1503 [-1507, -1499] | n=5 |
| E: return g x2.0 | legacy_latent_ode | adapt -1036 [-1190, -882]; no-adapt -1510 [-1513, -1506] | n=5 |
| E: return l x0.7 | physics | adapt -245 [-266, -223]; no-adapt -257 [-280, -233] | n=5 |
| E: return l x0.7 | physics_residual | adapt -245 [-267, -223]; no-adapt -268 [-316, -231] | n=5 |
| E: return l x0.7 | neural_hamiltonian | adapt -246 [-268, -224]; no-adapt -256 [-277, -234] | n=5 |
| E: return l x0.7 | mlp_ensemble | adapt -249 [-270, -228]; no-adapt -279 [-339, -232] | n=5 |
| E: return l x0.7 | neural_ode | adapt -248 [-273, -223]; no-adapt -279 [-346, -232] | n=5 |
| E: return l x0.7 | legacy_latent_ode | adapt -245 [-270, -225]; no-adapt -293 [-353, -246] | n=5 |
| E: return l x1.5 | physics | adapt -585 [-589, -582]; no-adapt -1049 [-1072, -1025] | n=5 |
| E: return l x1.5 | physics_residual | adapt -585 [-589, -582]; no-adapt -1003 [-1072, -920] | n=5 |
| E: return l x1.5 | neural_hamiltonian | adapt -587 [-593, -582]; no-adapt -972 [-1074, -864] | n=5 |
| E: return l x1.5 | mlp_ensemble | adapt -584 [-590, -577]; no-adapt -823 [-975, -674] | n=5 |
| E: return l x1.5 | neural_ode | adapt -580 [-584, -576]; no-adapt -1003 [-1078, -908] | n=5 |
| E: return l x1.5 | legacy_latent_ode | adapt -591 [-605, -581]; no-adapt -940 [-1021, -859] | n=5 |
| E: return m x0.5 | physics | adapt -210 [-212, -208]; no-adapt -237 [-240, -235] | n=5 |
| E: return m x0.5 | physics_residual | adapt -211 [-212, -209]; no-adapt -237 [-240, -234] | n=5 |
| E: return m x0.5 | neural_hamiltonian | adapt -213 [-216, -210]; no-adapt -237 [-240, -235] | n=5 |
| E: return m x0.5 | mlp_ensemble | adapt -217 [-221, -212]; no-adapt -237 [-240, -235] | n=5 |
| E: return m x0.5 | neural_ode | adapt -217 [-221, -213]; no-adapt -236 [-239, -234] | n=5 |
| E: return m x0.5 | legacy_latent_ode | adapt -221 [-229, -215]; no-adapt -237 [-240, -234] | n=5 |
| E: return m x2.0 | physics | adapt -639 [-649, -632]; no-adapt -861 [-893, -828] | n=5 |
| E: return m x2.0 | physics_residual | adapt -639 [-649, -632]; no-adapt -830 [-861, -794] | n=5 |
| E: return m x2.0 | neural_hamiltonian | adapt -652 [-674, -633]; no-adapt -837 [-870, -804] | n=5 |
| E: return m x2.0 | mlp_ensemble | adapt -663 [-683, -642]; no-adapt -831 [-872, -788] | n=5 |
| E: return m x2.0 | neural_ode | adapt -652 [-671, -636]; no-adapt -870 [-902, -824] | n=5 |
| E: return m x2.0 | legacy_latent_ode | adapt -692 [-759, -637]; no-adapt -856 [-890, -828] | n=5 |
| E: rel. error of a / c after 50 transitions | physics | median 1.9e-10 / 4.6e-10; max 5.5e-10 / 2.2e-09 | n=35 |
| F: return (last 3 eps), real env | oracle | -640 [-651, -629] | IQM -638, n=5 |
| F: return (last 3 eps), real env | physics | -631 [-643, -623] | IQM -628, n=5 |
| F: H=25 error, real env | physics | 8.06e-02 [3.25e-02, 1.29e-01] | n=5 |
| F: return (last 3 eps), real env | physics_nofric_residual | -639 [-650, -629] | IQM -639, n=5 |
| F: H=25 error, real env | physics_nofric_residual | 6.21e-03 [4.08e-03, 8.52e-03] | n=5 |
| F: return (last 3 eps), real env | physics_nofric | -673 [-724, -625] | IQM -662, n=5 |
| F: H=25 error, real env | physics_nofric | 2.79e-01 [2.76e-01, 2.81e-01] | n=5 |
| F: return (last 3 eps), real env | neural_hamiltonian | -638 [-656, -619] | IQM -637, n=5 |
| F: H=25 error, real env | neural_hamiltonian | 2.65e-01 [2.55e-01, 2.77e-01] | n=5 |
| F: return (last 3 eps), real env | mlp_ensemble | -638 [-646, -630] | IQM -636, n=5 |
| F: H=25 error, real env | mlp_ensemble | 7.09e-03 [6.30e-03, 7.87e-03] | n=5 |
| F: return (last 3 eps), real env | neural_ode | -644 [-654, -633] | IQM -645, n=5 |
| F: H=25 error, real env | neural_ode | 8.63e-03 [7.13e-03, 1.04e-02] | n=5 |
