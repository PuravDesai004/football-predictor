# FPL v3 Data Quality Audit

Phase: Tier 3 FPL Phase 1C

This audit is read-only against the database. It does not train models, tune metrics, or modify Tier 2 tables.

## Holdout Warning

2025-26 is available as raw imported history only. Do not use it for model selection or tuning; reserve it for final FPL evaluation if selected as holdout.

## Global Counts

| metric | value |
| --- | --- |
| max_season | 2025-26 |
| min_season | 2016-17 |
| max_gameweek | 47 |
| min_gameweek | 1 |
| total_rows | 253890 |
| seasons | 10 |
| distinct_player_name | 4342 |
| distinct_source_file | 379 |
| distinct_player_source_id | 866 |

## Season Coverage

| season | rows | distinct_players | min_gameweek | max_gameweek | source_files | rows_minutes_not_null | rows_total_points_not_null | rows_team_name_not_null | rows_position_not_null | rows_expected_goals_not_null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016-17 | 23679 | 683 | 1 | 38 | 38 | 23679 | 23679 | 0 | 0 | 0 |
| 2017-18 | 22467 | 647 | 1 | 38 | 38 | 22467 | 22467 | 0 | 0 | 0 |
| 2018-19 | 21790 | 625 | 1 | 38 | 38 | 21790 | 21790 | 0 | 0 | 0 |
| 2019-20 | 22560 | 666 | 1 | 47 | 38 | 22560 | 22560 | 0 | 0 | 0 |
| 2020-21 | 24365 | 712 | 1 | 38 | 38 | 24365 | 24365 | 24365 | 24365 | 0 |
| 2021-22 | 25447 | 735 | 1 | 38 | 38 | 25447 | 25447 | 25447 | 25447 | 0 |
| 2022-23 | 26505 | 777 | 1 | 38 | 37 | 26505 | 26505 | 26505 | 26505 | 26505 |
| 2023-24 | 29725 | 869 | 1 | 38 | 38 | 29725 | 29725 | 29725 | 29725 | 29725 |
| 2024-25 | 27605 | 805 | 1 | 38 | 38 | 27605 | 27605 | 27605 | 27605 | 27605 |
| 2025-26 | 29747 | 841 | 1 | 38 | 38 | 29747 | 29747 | 29747 | 29747 | 29747 |

## Required Column Audit

| column_name | null_count | null_rate |
| --- | --- | --- |
| season | 0 | 0.0 |
| gameweek | 0 | 0.0 |
| player_name | 0 | 0.0 |
| source_file | 0 | 0.0 |
| total_points | 0 | 0.0 |
| minutes | 0 | 0.0 |

## Optional Column Audit

| column_name | null_count | null_rate |
| --- | --- | --- |
| expected_assists | 140308 | 0.552633 |
| expected_goal_involvements | 140308 | 0.552633 |
| expected_goals | 140308 | 0.552633 |
| expected_goals_conceded | 140308 | 0.552633 |
| starts | 140308 | 0.552633 |
| position | 90496 | 0.356438 |
| team_name | 90496 | 0.356438 |
| creativity | 0 | 0.0 |
| fixture_id | 0 | 0.0 |
| ict_index | 0 | 0.0 |
| influence | 0 | 0.0 |
| kickoff_time | 0 | 0.0 |
| opponent_team_name | 0 | 0.0 |
| player_source_id | 0 | 0.0 |
| selected | 0 | 0.0 |
| threat | 0 | 0.0 |
| transfers_in | 0 | 0.0 |
| transfers_out | 0 | 0.0 |
| value | 0 | 0.0 |

## Team And Position Coverage

| audit_section | season | player_name | rows | team_name_null_rate | position_null_rate |
| --- | --- | --- | --- | --- | --- |
| season_coverage | 2016-17 |  | 23679 | 1.0 | 1.0 |
| season_coverage | 2017-18 |  | 22467 | 1.0 | 1.0 |
| season_coverage | 2018-19 |  | 21790 | 1.0 | 1.0 |
| season_coverage | 2019-20 |  | 22560 | 1.0 | 1.0 |
| season_coverage | 2020-21 |  | 24365 | 0.0 | 0.0 |
| season_coverage | 2021-22 |  | 25447 | 0.0 | 0.0 |
| season_coverage | 2022-23 |  | 26505 | 0.0 | 0.0 |
| season_coverage | 2023-24 |  | 29725 | 0.0 | 0.0 |
| season_coverage | 2024-25 |  | 27605 | 0.0 | 0.0 |
| season_coverage | 2025-26 |  | 29747 | 0.0 | 0.0 |
| top_missing_team_name | 2019-20 | Ainsley_Maitland-Niles_4 | 39 |  |  |
| top_missing_team_name | 2019-20 | Alexandre_Lacazette_12 | 39 |  |  |
| top_missing_team_name | 2019-20 | Aymeric_Laporte_202 | 39 |  |  |
| top_missing_team_name | 2019-20 | Benjamin_Mendy_204 | 39 |  |  |
| top_missing_team_name | 2019-20 | Bernardo Mota_Veiga de Carvalho e Silva_218 | 39 |  |  |
| top_missing_team_name | 2019-20 | Bernd_Leno_14 | 39 |  |  |
| top_missing_team_name | 2019-20 | Calum_Chambers_467 | 39 |  |  |
| top_missing_team_name | 2019-20 | Carl_Jenkinson_9 | 39 |  |  |
| top_missing_team_name | 2019-20 | Cédric_Soares_486 | 39 |  |  |
| top_missing_team_name | 2019-20 | Claudio_Bravo_213 | 39 |  |  |
| top_missing_team_name | 2019-20 | Daniel_Ceballos Fernández_469 | 39 |  |  |
| top_missing_team_name | 2019-20 | Danilo Luiz_da Silva_209 | 39 |  |  |
| top_missing_team_name | 2019-20 | David_Luiz Moreira Marinho_106 | 39 |  |  |
| top_missing_team_name | 2019-20 | David_Silva_219 | 39 |  |  |
| top_missing_team_name | 2019-20 | Ederson_Santana de Moraes_212 | 39 |  |  |
| top_missing_team_name | 2019-20 | Edward_Nketiah_13 | 39 |  |  |
| top_missing_team_name | 2019-20 | Emiliano_Martínez_427 | 39 |  |  |
| top_missing_team_name | 2019-20 | Fernando_Luiz Rosa_221 | 39 |  |  |
| top_missing_team_name | 2019-20 | Gabriel Fernando_de Jesus_211 | 39 |  |  |
| top_missing_team_name | 2019-20 | Gabriel Teodoro_Martinelli Silva_504 | 39 |  |  |
| top_missing_team_name | 2019-20 | Granit_Xhaka_18 | 39 |  |  |
| top_missing_team_name | 2019-20 | Héctor_Bellerín_2 | 39 |  |  |
| top_missing_team_name | 2019-20 | Henrikh_Mkhitaryan_16 | 39 |  |  |
| top_missing_team_name | 2019-20 | Ilkay_Gündogan_222 | 39 |  |  |
| top_missing_team_name | 2019-20 | João Pedro Cavaco_Cancelo_518 | 39 |  |  |
| top_missing_team_name | 2019-20 | John_Stones_207 | 39 |  |  |
| top_missing_team_name | 2019-20 | José Ángel_Esmorís Tasende_440 | 39 |  |  |
| top_missing_team_name | 2019-20 | Joseph_Willock_490 | 39 |  |  |
| top_missing_team_name | 2019-20 | Kevin_De Bruyne_215 | 39 |  |  |
| top_missing_team_name | 2019-20 | Kieran_Tierney_515 | 39 |  |  |
| top_missing_position | 2019-20 | Ainsley_Maitland-Niles_4 | 39 |  |  |
| top_missing_position | 2019-20 | Alexandre_Lacazette_12 | 39 |  |  |
| top_missing_position | 2019-20 | Aymeric_Laporte_202 | 39 |  |  |
| top_missing_position | 2019-20 | Benjamin_Mendy_204 | 39 |  |  |
| top_missing_position | 2019-20 | Bernardo Mota_Veiga de Carvalho e Silva_218 | 39 |  |  |
| top_missing_position | 2019-20 | Bernd_Leno_14 | 39 |  |  |
| top_missing_position | 2019-20 | Calum_Chambers_467 | 39 |  |  |
| top_missing_position | 2019-20 | Carl_Jenkinson_9 | 39 |  |  |
| top_missing_position | 2019-20 | Cédric_Soares_486 | 39 |  |  |
| top_missing_position | 2019-20 | Claudio_Bravo_213 | 39 |  |  |
| top_missing_position | 2019-20 | Daniel_Ceballos Fernández_469 | 39 |  |  |
| top_missing_position | 2019-20 | Danilo Luiz_da Silva_209 | 39 |  |  |
| top_missing_position | 2019-20 | David_Luiz Moreira Marinho_106 | 39 |  |  |
| top_missing_position | 2019-20 | David_Silva_219 | 39 |  |  |
| top_missing_position | 2019-20 | Ederson_Santana de Moraes_212 | 39 |  |  |
| top_missing_position | 2019-20 | Edward_Nketiah_13 | 39 |  |  |
| top_missing_position | 2019-20 | Emiliano_Martínez_427 | 39 |  |  |
| top_missing_position | 2019-20 | Fernando_Luiz Rosa_221 | 39 |  |  |
| top_missing_position | 2019-20 | Gabriel Fernando_de Jesus_211 | 39 |  |  |
| top_missing_position | 2019-20 | Gabriel Teodoro_Martinelli Silva_504 | 39 |  |  |
| top_missing_position | 2019-20 | Granit_Xhaka_18 | 39 |  |  |
| top_missing_position | 2019-20 | Héctor_Bellerín_2 | 39 |  |  |
| top_missing_position | 2019-20 | Henrikh_Mkhitaryan_16 | 39 |  |  |
| top_missing_position | 2019-20 | Ilkay_Gündogan_222 | 39 |  |  |
| top_missing_position | 2019-20 | João Pedro Cavaco_Cancelo_518 | 39 |  |  |
| top_missing_position | 2019-20 | John_Stones_207 | 39 |  |  |
| top_missing_position | 2019-20 | José Ángel_Esmorís Tasende_440 | 39 |  |  |
| top_missing_position | 2019-20 | Joseph_Willock_490 | 39 |  |  |
| top_missing_position | 2019-20 | Kevin_De Bruyne_215 | 39 |  |  |
| top_missing_position | 2019-20 | Kieran_Tierney_515 | 39 |  |  |

## Player Identity Stability

Create fpl_player_identity_map_v3 before modeling. player_source_id should be treated as season-local, not globally stable.

| audit_type | season | player_name | player_source_id | distinct_id_count | rows | examples |
| --- | --- | --- | --- | --- | --- | --- |
| same_name_multiple_ids_within_season | 2021-22 | Ben Davies |  | 2 | 76 | 248, 364 |
| same_name_multiple_ids_within_season | 2022-23 | Ben Davies |  | 2 | 76 | 432, 499 |
| same_name_multiple_ids_within_season | 2020-21 | Ben Davies |  | 2 | 55 | 395, 653 |
| same_name_multiple_ids_within_season | 2021-22 | Álvaro Fernández |  | 2 | 42 | 556, 728 |
| same_id_multiple_names_within_season | 2018-19 |  | 515 | 2 | 38 | Caglar_Söyüncü_515, Çaglar_Söyüncü_515 |
| same_id_multiple_names_within_season | 2023-24 |  | 120 | 2 | 38 | Yegor Yarmoliuk, Yegor Yarmolyuk |
| same_id_multiple_names_within_season | 2023-24 |  | 620 | 2 | 38 | Michael Olakigbe, Michale Olakigbe |
| same_id_multiple_names_within_season | 2023-24 |  | 687 | 2 | 35 | Djordje Petrovic, Đorđe Petrović |
| same_id_multiple_names_within_season | 2023-24 |  | 802 | 2 | 18 | Max Kinsey, Max Kinsey-Wellings |
| same_id_multiple_names_within_season | 2024-25 |  | 748 | 2 | 16 | Ivan Juric, Simon Rusk |
| same_id_multiple_names_global |  |  | 120 | 11 | 380 | 2016-17:Andros_Townsend, 2017-18:Martin_Kelly, 2018-19:Davide_Zappacosta_120, 2019-20:Daniel_Drinkwater_120, 2020-21:Mason Mount, 2021-22:Olivier Giroud, 2022-23:Moisés Caicedo Corozo, 2023-24:Yegor Yarmoliuk, 2023-24:Yegor Yarmolyuk, 2024-25:Lewis Dunk, 2025-26:Kevin Schade |
| same_id_multiple_names_global |  |  | 515 | 11 | 380 | 2016-17:Josh_Clackstone, 2017-18:Sullay_Kaikai, 2018-19:Caglar_Söyüncü_515, 2018-19:Çaglar_Söyüncü_515, 2019-20:Kieran_Tierney_515, 2020-21:Vitor Ferreira, 2021-22:Bali Mumba, 2022-23:Keane Lewis-Potter, 2023-24:Oliver Skipp, 2024-25:Maxwel Cornet, 2025-26:Morgan Gibbs-White |
| same_id_multiple_names_global |  |  | 620 | 11 | 245 | 2016-17:Michael_Folivi, 2017-18:Alexander_Sørloth, 2018-19:Matt_Butcher_620, 2019-20:Josh_Brownhill_620, 2020-21:Charlie Cresswell, 2021-22:Tyler Morton, 2022-23:Arthur Henrique Ramos de Oliveira Melo, 2023-24:Michael Olakigbe, 2023-24:Michale Olakigbe, 2024-25:Ramón Sosa, 2025-26:George Earthy |
| same_id_multiple_names_global |  |  | 106 | 10 | 382 | 2016-17:Martin_Kelly, 2017-18:Willian_Borges Da Silva, 2018-19:Josh_Murphy_106, 2019-20:David_Luiz Moreira Marinho_106, 2020-21:Ross Barkley, 2021-22:Ben Mee, 2022-23:Lewis Dunk, 2023-24:Mathias Jensen, 2024-25:Kevin Schade, 2025-26:Nathan Collins |
| same_id_multiple_names_global |  |  | 12 | 10 | 382 | 2016-17:Alexis_Sánchez, 2017-18:Shkodran_Mustafi, 2018-19:Sokratis_Papastathopoulos_12, 2019-20:Alexandre_Lacazette_12, 2020-21:Emiliano Martínez, 2021-22:Mohamed Naser El Sayed Elneny, 2022-23:Emile Smith Rowe, 2023-24:Gabriel Martinelli Silva, 2024-25:Ethan Nwaneri, 2025-26:Oleksandr Zinchenko |
| same_id_multiple_names_global |  |  | 282 | 10 | 382 | 2016-17:Stewart_Downing, 2017-18:Paul_Pogba, 2018-19:David_De Gea_282, 2019-20:Mario_Vrancic_282, 2020-21:Gabriel Fernando de Jesus, 2021-22:Donny van de Beek, 2022-23:Fabio Henrique Tavares, 2023-24:Harrison Reed, 2024-25:Axel Tuanzebe, 2025-26:Franco Umeh-Chibueze |
| same_id_multiple_names_global |  |  | 13 | 10 | 381 | 2016-17:Theo_Walcott, 2017-18:Sead_Kolasinac, 2018-19:Mesut_Özil_13, 2019-20:Edward_Nketiah_13, 2020-21:Calum Chambers, 2021-22:Ainsley Maitland-Niles, 2022-23:Bukayo Saka, 2023-24:Eddie Nketiah, 2024-25:Martin Ødegaard, 2025-26:Brayden Clarke |
| same_id_multiple_names_global |  |  | 130 | 10 | 381 | 2016-17:Tyias_Browning, 2017-18:Bakary_Sako, 2018-19:Ethan_Ampadu_130, 2019-20:Connor_Wickham_130, 2020-21:James McArthur, 2021-22:N'Golo Kanté, 2022-23:Jorge Luiz Frello Filho, 2023-24:Julio Enciso, 2024-25:Kacper Kozłowski, 2025-26:Romelle Donovan |
| same_id_multiple_names_global |  |  | 14 | 10 | 381 | 2016-17:Mesut_Özil, 2017-18:Alexis_Sánchez, 2018-19:Aaron_Ramsey_14, 2019-20:Bernd_Leno_14, 2020-21:Sead Kolasinac, 2021-22:Rob Holding, 2022-23:Takehiro Tomiyasu, 2023-24:Martin Ødegaard, 2024-25:Aaron Ramsdale, 2025-26:Maldini Kacurri |
| same_id_multiple_names_global |  |  | 16 | 10 | 381 | 2016-17:Aaron_Ramsey, 2017-18:Mesut_Özil, 2018-19:Mohamed_Elneny_16, 2019-20:Henrikh_Mkhitaryan_16, 2020-21:Rob Holding, 2021-22:Kieran Tierney, 2022-23:Gabriel dos Santos Magalhães, 2023-24:Nicolas Pépé, 2024-25:Declan Rice, 2025-26:Bukayo Saka |
| same_id_multiple_names_global |  |  | 202 | 10 | 381 | 2016-17:Joe_Allen, 2017-18:Daniel_Drinkwater, 2018-19:Philip_Billing_202, 2019-20:Aymeric_Laporte_202, 2020-21:Patrick Bamford, 2021-22:Marc Albrighton, 2022-23:Anthony Knockaert, 2023-24:Conor Gallagher, 2024-25:Rob Holding, 2025-26:Jacob Bruun Larsen |
| same_id_multiple_names_global |  |  | 203 | 10 | 381 | 2016-17:Philippe_Coutinho, 2017-18:Demarai_Gray, 2018-19:Tom_Ince_203, 2019-20:Kyle_Walker_203, 2020-21:Jack Harrison, 2021-22:Nampalys Mendy, 2022-23:Nathaniel Chalobah, 2023-24:Malo Gusto, 2024-25:Will Hughes, 2025-26:Enock Agyei |
| same_id_multiple_names_global |  |  | 204 | 10 | 381 | 2016-17:Emre_Can, 2017-18:Daniel_Amartey, 2018-19:Danny_Williams_204, 2019-20:Benjamin_Mendy_204, 2020-21:Kalvin Phillips, 2021-22:Danny Ward, 2022-23:Neeskens Kebano, 2023-24:Lewis Hall, 2024-25:Sam Johnstone, 2025-26:Darko Churlinov |
| same_id_multiple_names_global |  |  | 206 | 10 | 381 | 2016-17:Lazar_Markovic, 2017-18:Wilfred_Ndidi, 2018-19:Alex_Pritchard_206, 2019-20:Oleksandr_Zinchenko_206, 2020-21:Tyler Roberts, 2021-22:Dennis Praet, 2022-23:Joe Bryan, 2023-24:Reece James, 2024-25:Jefferson Lerma Solís, 2025-26:Marcus Edwards |
| same_id_multiple_names_global |  |  | 207 | 10 | 381 | 2016-17:Oluwaseyi_Ojo, 2017-18:Ahmed_Musa, 2018-19:Ramadan_Sobhi_207, 2019-20:John_Stones_207, 2020-21:Jay-Roy Grot, 2021-22:Ricardo Domingos Barbosa Pereira, 2022-23:Paulo Gazzaniga Farias, 2023-24:Romelu Lukaku Bolingoli, 2024-25:Jean-Philippe Mateta, 2025-26:Hannibal Mejbri |
| same_id_multiple_names_global |  |  | 208 | 10 | 381 | 2016-17:James_Milner, 2017-18:Vicente_Iborra, 2018-19:Juninho_Bacuna_208, 2019-20:Nicolás_Otamendi_208, 2020-21:Kamil Miazek, 2021-22:Daniel Amartey, 2022-23:Terence Kongolo, 2023-24:Noni Madueke, 2024-25:Matheus França de Oliveira, 2025-26:Luca Koleosho |
| same_id_multiple_names_global |  |  | 209 | 10 | 381 | 2016-17:Roberto_Firmino, 2017-18:Jamie_Vardy, 2018-19:Collin_Quaner_209, 2019-20:Danilo Luiz_da Silva_209, 2020-21:Ian Carlo Poveda-Ocampo, 2021-22:Timothy Castagne, 2022-23:Tosin Adarabioyo, 2023-24:Mason Mount, 2024-25:Remi Matthews, 2025-26:Josh Laurent |
| same_id_multiple_names_global |  |  | 210 | 10 | 381 | 2016-17:Cameron_Brannagan, 2017-18:Leonardo_Ulloa, 2018-19:Laurent_Depoitre_210, 2019-20:Sergio_Agüero_210, 2020-21:Jamie Shackleton, 2021-22:Youri Tielemans, 2022-23:Aleksandar Mitrović, 2023-24:Mykhailo Mudryk, 2024-25:Tyrick Mitchell, 2025-26:Loum Tchaouna |
| same_id_multiple_names_global |  |  | 211 | 10 | 381 | 2016-17:Pedro_Chirivella, 2017-18:Shinji_Okazaki, 2018-19:Steve_Mounie_211, 2019-20:Gabriel Fernando_de Jesus_211, 2020-21:Pascal Struijk, 2021-22:Ayoze Pérez, 2022-23:Harrison Reed, 2023-24:Nicolas Jackson, 2024-25:Daniel Muñoz, 2025-26:Aaron Ramsey |
| same_id_multiple_names_global |  |  | 212 | 10 | 381 | 2016-17:Sadio_Mané, 2017-18:Islam_Slimani, 2018-19:Elias_Kachunga_212, 2019-20:Ederson_Santana de Moraes_212, 2020-21:Jordan Stevens, 2021-22:James Maddison, 2022-23:Harry Wilson, 2023-24:Christopher Nkunku, 2024-25:David Ozoh, 2025-26:Oluwaseun Adewumi |
| same_id_multiple_names_global |  |  | 213 | 10 | 381 | 2016-17:Christian_Benteke, 2017-18:Simon_Mignolet, 2018-19:Kasper_Schmeichel_213, 2019-20:Claudio_Bravo_213, 2020-21:Illan Meslier, 2021-22:Kelechi Iheanacho, 2022-23:Marek Rodák, 2023-24:Christian Pulisic, 2024-25:Jesurun Rak-Sakyi, 2025-26:Jaydon Banel |
| same_id_multiple_names_global |  |  | 214 | 10 | 381 | 2016-17:Daniel_Sturridge, 2017-18:Loris_Karius, 2018-19:Eldin_Jakupovic_214, 2019-20:Raheem_Sterling_214, 2020-21:Leif Davis, 2021-22:Hamza Choudhury, 2022-23:Ivan Neves Abreu Cavaleiro, 2023-24:Malang Sarr, 2024-25:Jeffrey Schlupp, 2025-26:Mike Trésor Ndayishimiye |
| same_id_multiple_names_global |  |  | 215 | 10 | 381 | 2016-17:Mario_Balotelli, 2017-18:Dejan_Lovren, 2018-19:Wes_Morgan_215, 2019-20:Kevin_De Bruyne_215, 2020-21:Oliver Casey, 2021-22:Harvey Barnes, 2022-23:Kenny Tete, 2023-24:Gabriel Słonina, 2024-25:Joel Ward, 2025-26:Zian Flemming |
| same_id_multiple_names_global |  |  | 216 | 10 | 381 | 2016-17:Danny_Ings, 2017-18:Alberto_Moreno, 2018-19:Christian_Fuchs_216, 2019-20:Leroy_Sané_216, 2020-21:Wes Morgan, 2021-22:Wilfred Ndidi, 2022-23:Josh Onomah, 2023-24:Raheem Sterling, 2024-25:Adam Wharton, 2025-26:Zeki Amdouni |
| same_id_multiple_names_global |  |  | 217 | 10 | 381 | 2016-17:Divock_Origi, 2017-18:Nathaniel_Clyne, 2018-19:Danny_Simpson_217, 2019-20:Riyad_Mahrez_217, 2020-21:Kasper Schmeichel, 2021-22:Çaglar Söyüncü, 2022-23:Antonee Robinson, 2023-24:Thiago Emiliano da Silva, 2024-25:Abdoulaye Doucouré, 2025-26:Lyle Foster |
| same_id_multiple_names_global |  |  | 218 | 10 | 381 | 2016-17:Joe_Hart, 2017-18:Joseph_Gomez, 2018-19:Yohan_Benalouane_218, 2019-20:Bernardo Mota_Veiga de Carvalho e Silva_218, 2020-21:Christian Fuchs, 2021-22:James Justin, 2022-23:Rodrigo Muniz Carvalho, 2023-24:Hakim Ziyech, 2024-25:Norberto Bercique Gomes Betuncal, 2025-26:Ashley Barnes |
| same_id_multiple_names_global |  |  | 219 | 10 | 381 | 2016-17:Willy_Caballero, 2017-18:Joel_Matip, 2018-19:Benjamin_Chilwell_219, 2019-20:David_Silva_219, 2020-21:Jonny Evans, 2021-22:Luke Thomas, 2022-23:Jay Stansfield, 2023-24:Naouirou Ahamada, 2024-25:Jarrad Branthwaite, 2025-26:Michael Obafemi |
| same_id_multiple_names_global |  |  | 220 | 10 | 381 | 2016-17:Aleksandar_Kolarov, 2017-18:Ragnar_Klavan, 2018-19:Harry_Maguire_220, 2019-20:Phil_Foden_220, 2020-21:Adrien Silva, 2021-22:Wesley Fofana, 2022-23:João Palhinha Gonçalves, 2023-24:Joachim Andersen, 2024-25:Dominic Calvert-Lewin, 2025-26:Robert Lynch Sánchez |
| same_id_multiple_names_global |  |  | 221 | 10 | 381 | 2016-17:Pablo_Zabaleta, 2017-18:Mamadou_Sakho, 2018-19:Ricardo Domingos_Barbosa Pereira_221, 2019-20:Fernando_Luiz Rosa_221, 2020-21:Marc Albrighton, 2021-22:James Milner, 2022-23:Liam Cooper, 2023-24:Jordan Ayew, 2024-25:Séamus Coleman, 2025-26:Filip Jörgensen |
| same_id_multiple_names_global |  |  | 222 | 10 | 381 | 2016-17:Vincent_Kompany, 2017-18:James_Milner, 2018-19:Jonny_Evans_222, 2019-20:Ilkay_Gündogan_222, 2020-21:Matty James, 2021-22:Jordan Henderson, 2022-23:Luke Ayling, 2023-24:Nathaniel Clyne, 2024-25:Idrissa Gueye, 2025-26:Mike Penders |

## Duplicate And Anomaly Checks

### Duplicate Player Checks

| duplicate_key | season | gameweek | player_name | player_source_id | fixture_id | source_file | rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| name_fixture_source_file | 2021-22 | 18 | Ben Davies |  | 177 | data/vaastav_fpl_history/data/2021-22/gws/gw18.csv | 2 |
| name_fixture_source_file | 2021-22 | 35 | Álvaro Fernández |  | 344 | data/vaastav_fpl_history/data/2021-22/gws/gw35.csv | 2 |
| name_fixture_source_file | 2021-22 | 36 | Ben Davies |  | 358 | data/vaastav_fpl_history/data/2021-22/gws/gw36.csv | 2 |
| name_fixture_source_file | 2022-23 | 15 | Ben Davies |  | 148 | data/vaastav_fpl_history/data/2022-23/gws/gw15.csv | 2 |
| name_fixture_source_file | 2022-23 | 34 | Ben Davies |  | 338 | data/vaastav_fpl_history/data/2022-23/gws/gw34.csv | 2 |

### Gameweek And Value Anomalies

| check_name | count |
| --- | --- |
| gameweek_outside_1_to_47 | 0 |
| seasons_more_than_38_unique_gameweeks | 0 |
| minutes_less_than_zero | 0 |
| total_points_null | 0 |
| total_points_gt_30 | 0 |
| value_lte_zero_not_null | 0 |

## Training Readiness

| season | status | rows | source_files | points_not_null_rate | minutes_not_null_rate | team_name_not_null_rate | position_not_null_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2016-17 | PARTIAL | 23679 | 38 | 1.0 | 1.0 | 0.0 | 0.0 | team_name coverage below 95%; position coverage below 95% |
| 2017-18 | PARTIAL | 22467 | 38 | 1.0 | 1.0 | 0.0 | 0.0 | team_name coverage below 95%; position coverage below 95% |
| 2018-19 | PARTIAL | 21790 | 38 | 1.0 | 1.0 | 0.0 | 0.0 | team_name coverage below 95%; position coverage below 95% |
| 2019-20 | PARTIAL | 22560 | 38 | 1.0 | 1.0 | 0.0 | 0.0 | team_name coverage below 95%; position coverage below 95% |
| 2020-21 | READY | 24365 | 38 | 1.0 | 1.0 | 1.0 | 1.0 | ready for modeling after feature build |
| 2021-22 | READY | 25447 | 38 | 1.0 | 1.0 | 1.0 | 1.0 | ready for modeling after feature build |
| 2022-23 | READY | 26505 | 37 | 1.0 | 1.0 | 1.0 | 1.0 | ready for modeling after feature build |
| 2023-24 | READY | 29725 | 38 | 1.0 | 1.0 | 1.0 | 1.0 | ready for modeling after feature build |
| 2024-25 | READY | 27605 | 38 | 1.0 | 1.0 | 1.0 | 1.0 | ready for modeling after feature build |
| 2025-26 | READY | 29747 | 38 | 1.0 | 1.0 | 1.0 | 1.0 | reserved: do not use for model selection/tuning |

## Protected Tier 2 Table Safety

| table | before | after |
| --- | ---: | ---: |
| players | 841 | 841 |
| teams | 20 | 20 |
| fixtures | 380 | 380 |
| gameweeks | 38 | 38 |
| player_gameweek_history | 29747 | 29747 |
| player_gameweek_features | 29747 | 29747 |

`models/saved` git status: `clean`
