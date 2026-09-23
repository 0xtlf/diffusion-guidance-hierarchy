# Superseded runs (summaries preserved, directories removed)

Archived 2026-09-23. Kept runs live in `runs/`; these were exploratory
or superseded and their outputs were regenerable from a config plus a seed.


## alpha-sweep-klein-0901-155232

_no summary.json (partial or training-only run)_

## alpha-sweep-sphere-40k-0902-174430

_no summary.json (partial or training-only run)_

## cond-uniform

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.1288000000000017,
    "ratio_to_floor": 1.3145594952936563,
    "ks_p_vs_uniform": 0.2853678644283423,
    "ks_p_vs_restriction": 0.0
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 1.3687999999999958,
    "ratio_to_floor": 13.97025649967335,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 1.0
  },
  {
    "name": "tempered, alpha=0.5",
    "max_deviation": 0.2727999999999994,
    "ratio_to_floor": 2.7842533409635397,
    "ks_p_vs_uniform": 6.012927839854377e-22,
    "ks_p_vs_restriction": 5.101006386991155e-290
  }
]
```

## cond-uniform-alpha

_no summary.json (partial or training-only run)_

## klein-a06

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.13120000000000076,
    "ratio_to_floor": 1.3390543927214786,
    "ks_p_vs_uniform": 0.3634363388061881,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 6.0806359406041225e-18,
    "abs_constraint": 1.043248820664644e-16
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 1.9712000000000036,
    "ratio_to_floor": 20.118475754059208,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 1.0,
    "dist_M": 4.7402996339350166e-18,
    "abs_constraint": 1.6863732632543816e-16
  },
  {
    "name": "untempered chain (alpha=0)",
    "max_deviation": 1.9592000000000036,
    "ratio_to_floor": 19.99600126692005,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 0.9999999655622609,
    "dist_M": 0.012480368718471883,
    "abs_constraint": 0.007976117968808424
  },
  {
    "name": "tempered, alpha=0.6",
    "max_deviation": 1.2127999999999934,
    "ratio_to_floor": 12.378088166864261,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 5.678251352249618e-38,
    "dist_M": 0.05353922795690419,
    "abs_constraint": 0.03328649220784257
  }
]
```

## klein-a07-short

```json
[
  {
    "plane": 0,
    "offset": -0.21210802902534864,
    "length": 6.6615139601380955,
    "components": 1,
    "ptw_median": 0.7653495961174152,
    "steps": 4475,
    "ks_start": 0.22207390384661702,
    "ks_end": 0.023511128915863322,
    "ks_predicted": 0.009669720437657792,
    "ks_excess": 0.021430578376606817,
    "max_deviation": 0.21440000000000015,
    "ratio_to_floor": 2.1882108368863076,
    "efolds": 2.000031737506888
  },
  {
    "plane": 1,
    "offset": -0.6199440746872759,
    "length": 5.200749014903507,
    "components": 1,
    "ptw_median": 0.9302110005089117,
    "steps": 2728,
    "ks_start": 0.1370973433253937,
    "ks_end": 0.05321237311712501,
    "ks_predicted": 0.0065580438954086495,
    "ks_excess": 0.052806710871081734,
    "max_deviation": 0.2912000000000001,
    "ratio_to_floor": 2.9720475545769243,
    "efolds": 2.0003330355639592
  },
  {
    "plane": 2,
    "offset": -0.5885841269839045,
    "length": 5.236971753915168,
    "components": 1,
    "ptw_median": 0.8948014181466473,
    "steps": 2766,
    "ks_start": 0.14473963156607872,
    "ks_end": 0.032078799990598816,
    "ks_predicted": 0.006999872985097688,
    "ks_excess": 0.03130576922915874,
    "max_deviation": 0.20319999999999938,
    "ratio_to_floor": 2.073901315556418,
    "efolds": 2.000236947681868
  },
  {
    "plane": 3,
    "offset": 0.12282794128532754,
    "length": 7.836593344725344,
    "components": 1,
    "ptw_median": 0.832037541116888,
    "steps": 6193,
    "ks_start": 0.2862889957695377,
    "ks_end": 0.045056992815203145,
    "ks_predicted": 0.014257553342709783,
    "ks_excess": 0.042741721704080346,
    "max_deviation": 0.44000000000000017,
    "ratio_to_floor": 4.490731195102495,
    "efolds": 2.0000285278651933
  },
  {
    "plane": 4,
    "offset": 0.20123267923614574,
    "length": 6.825689012939516,
    "components": 1,
    "ptw_median": 0.8053521188495953,
    "steps": 4699,
    "ks_start": 0.1773115934160077,
    "ks_end": 0.07577651041899125,
    "ks_predicted": 0.011314390328372281,
    "ks_excess": 0.07492705854880949,
    "max_deviation": 1.0063999999999989,
    "ratio_to_floor": 10.271526988070782,
    "efolds": 2.0003324609060438
  }
]
```

## klein-a1

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.13120000000000076,
    "ratio_to_floor": 1.3390543927214786,
    "ks_p_vs_uniform": 0.3634363388061881,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 6.0806359406041225e-18,
    "abs_constraint": 1.043248820664644e-16
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 1.9712000000000036,
    "ratio_to_floor": 20.118475754059208,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 1.0,
    "dist_M": 4.7402996339350166e-18,
    "abs_constraint": 1.6863732632543816e-16
  },
  {
    "name": "untempered chain (alpha=0)",
    "max_deviation": 1.9592000000000036,
    "ratio_to_floor": 19.99600126692005,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 0.9999999655622609,
    "dist_M": 0.012480368718471883,
    "abs_constraint": 0.007976117968808424
  },
  {
    "name": "tempered, alpha=1",
    "max_deviation": 1.2392000000000025,
    "ratio_to_floor": 12.647532038570503,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 3.1576036214478855e-58,
    "dist_M": 0.13502408194948207,
    "abs_constraint": 0.08466545649429504
  }
]
```

## klein-a1-conn

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.12559999999999438,
    "ratio_to_floor": 1.2818996320564726,
    "ks_p_vs_uniform": 0.6692026067740229,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 5.581981520599845e-18,
    "abs_constraint": 5.664912983149861e-17
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 1.8632000000000088,
    "ratio_to_floor": 19.01620536980683,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 1.0,
    "dist_M": 5.206114313469121e-18,
    "abs_constraint": 7.842615445952106e-17
  },
  {
    "name": "untempered chain (alpha=0)",
    "max_deviation": 1.2535999999999978,
    "ratio_to_floor": 12.794501423137445,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 4.021313536619903e-261,
    "dist_M": 0.013605813160316331,
    "abs_constraint": 0.008382592453545362
  },
  {
    "name": "tempered, alpha=1",
    "max_deviation": 0.26239999999999997,
    "ratio_to_floor": 2.6781087854429413,
    "ks_p_vs_uniform": 4.2132117062519433e-22,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 0.13740840426505002,
    "abs_constraint": 0.08354874175583636
  }
]
```

## klein-conn2

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.12559999999999438,
    "ratio_to_floor": 1.2818996320564726,
    "ks_p_vs_uniform": 0.6692026067740229,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 5.581981520599845e-18,
    "abs_constraint": 5.664912983149861e-17
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 1.8632000000000088,
    "ratio_to_floor": 19.01620536980683,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 1.0,
    "dist_M": 5.206114313469121e-18,
    "abs_constraint": 7.842615445952106e-17
  },
  {
    "name": "untempered chain (alpha=0)",
    "max_deviation": 1.2535999999999978,
    "ratio_to_floor": 12.794501423137445,
    "ks_p_vs_uniform": 0.0,
    "ks_p_vs_restriction": 4.021313536619903e-261,
    "dist_M": 0.013605813160316331,
    "abs_constraint": 0.008382592453545362
  },
  {
    "name": "tempered, alpha=0.6",
    "max_deviation": 0.11920000000000008,
    "ratio_to_floor": 1.2165799055823128,
    "ks_p_vs_uniform": 9.792994263494093e-07,
    "ks_p_vs_restriction": 0.0,
    "dist_M": 0.05409316136575514,
    "abs_constraint": 0.03351922682048283
  }
]
```

## klein-connected

```json
[
  {
    "name": "exact uniform on N (target)",
    "max_deviation": 0.09279999999999766,
    "ratio_to_floor": 0.9471360338761384,
    "ks_p_vs_uniform": 0.03310089639367664,
    "ks_p_vs_restriction": 1.5662452017771063e-29,
    "dist_M": 6.203197259870434e-19,
    "abs_constraint": 1.103813974313093
  },
  {
    "name": "exact p_data|N (start)",
    "max_deviation": 0.4207999999999996,
    "ratio_to_floor": 4.294772015679835,
    "ks_p_vs_uniform": 5.920644746053866e-69,
    "ks_p_vs_restriction": 1.0,
    "dist_M": 7.974470206747466e-19,
    "abs_constraint": 1.103813974313093
  },
  {
    "name": "untempered chain (alpha=0)",
    "max_deviation": 0.37039999999999984,
    "ratio_to_floor": 3.78037916969537,
    "ks_p_vs_uniform": 1.5676274940743413e-70,
    "ks_p_vs_restriction": 0.9999999999870868,
    "dist_M": 0.012521971530455868,
    "abs_constraint": 1.1037060553522882
  },
  {
    "name": "tempered, alpha=0.6",
    "max_deviation": 0.5839999999999972,
    "ratio_to_floor": 5.960425040772372,
    "ks_p_vs_uniform": 4.5340082171157966e-159,
    "ks_p_vs_restriction": 4.486566856335323e-184,
    "dist_M": 0.05942389233977129,
    "abs_constraint": 1.1038267187998452
  },
  {
    "name": "tempered, alpha=1",
    "max_deviation": 0.7856000000000123,
    "ratio_to_floor": 8.017996424710395,
    "ks_p_vs_uniform": 8.056968001180238e-108,
    "ks_p_vs_restriction": 4.560846389822087e-139,
    "dist_M": 0.14548234099245658,
    "abs_constraint": 1.1043568749977002
  }
]
```
