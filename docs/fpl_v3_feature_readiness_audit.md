# FPL v3 Feature Readiness Audit

Generated: 2026-08-07 12:17:13 UTC

## Scope

Phase 2A builds chronological pre-gameweek player features only. No model training, evaluation, tuning, Streamlit changes, or production model artifact work was performed.

2025-26 is present only as raw feature output metadata and was not loaded into any model dataframe.

## Feature Definitions

| Feature | Source column | Lookback | Time cutoff | Null behavior |
|---------|---------------|----------|-------------|---------------|
| target_total_points | total_points | current row target | current gameweek target only | preserved null if source null |
| prior_points_last1 | total_points | last 1 prior player row | strictly before target row | null if no prior row |
| prior_points_last3/5/10 | total_points | mean over last N prior player rows | strictly before target row | null if no prior row |
| prior_points_season | total_points | same-season cumulative sum before target GW | gameweeks < current GW | 0 at first same-season GW |
| prior_minutes_last3/5/10 | minutes | mean over last N prior player rows | strictly before target row | null if no prior row |
| prior_appearances_last5 | minutes > 0 | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_starts_last5 | starts | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_goals_last5 | goals_scored | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_assists_last5 | assists | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_bonus_last5 | bonus | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_clean_sheets_last5 | clean_sheets | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_saves_last5 | saves | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_xg_last5 | expected_goals | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_xa_last5 | expected_assists | sum over last 5 prior player rows | strictly before target row | null if no prior row |
| prior_points_per_90 | total_points, minutes | all prior player rows | strictly before target row | null if no prior minutes |
| prior_minutes_total | minutes | all prior player rows | strictly before target row | 0 if no prior minutes |
| prior_gameweeks_played | minutes > 0 | all prior player rows | strictly before target row | 0 if no prior appearances |

No final-season players_raw.csv team or position metadata is used as a historical model feature.

## Global Results

| Metric | Value |
|--------|-------|
| Source history rows | 253890 |
| Identity-map rows | 7358 |
| Feature table rows | 244737 |
| Distinct fpl_code | 2643 |
| Rows with no identity mapping | 0 |
| Rows with no prior history | 2643 |
| Duplicate canonical_player_key/season/gameweek | 0 |
| Target null count | 0 |
| Target/result mismatch count | 0 |
| 2025-26 feature rows | 29338 |

## Leakage Audit

| Check | Result |
|-------|--------|
| Rows checked with prior history | 242094 |
| Future/current gameweek leak rows | 0 |
| target_total_points absent from model feature list | True |
| No future-season rows in prior windows | True |
| No current-gameweek source values in input features | True |
| No final-season metadata used as historical features | True |

## Rows By Season

| season | rows | distinct_fpl_code | no_prior_history_rows | target_null_rows |
| --- | --- | --- | --- | --- |
| 2016-17 | 23106 | 683 | 683 | 0 |
| 2017-18 | 21797 | 647 | 212 | 0 |
| 2018-19 | 21134 | 624 | 191 | 0 |
| 2019-20 | 22313 | 666 | 205 | 0 |
| 2020-21 | 22889 | 713 | 198 | 0 |
| 2021-22 | 23230 | 737 | 206 | 0 |
| 2022-23 | 24957 | 778 | 249 | 0 |
| 2023-24 | 28742 | 865 | 259 | 0 |
| 2024-25 | 27231 | 804 | 210 | 0 |
| 2025-26 | 29338 | 841 | 230 | 0 |

## Rows By Gameweek

| season | gameweek | rows |
| --- | --- | --- |
| 2016-17 | 1 | 524 |
| 2016-17 | 2 | 537 |
| 2016-17 | 3 | 548 |
| 2016-17 | 4 | 581 |
| 2016-17 | 5 | 584 |
| 2016-17 | 6 | 586 |
| 2016-17 | 7 | 589 |
| 2016-17 | 8 | 592 |
| 2016-17 | 9 | 594 |
| 2016-17 | 10 | 598 |
| 2016-17 | 11 | 599 |
| 2016-17 | 12 | 601 |
| 2016-17 | 13 | 604 |
| 2016-17 | 14 | 605 |
| 2016-17 | 15 | 610 |
| 2016-17 | 16 | 614 |
| 2016-17 | 17 | 616 |
| 2016-17 | 18 | 618 |
| 2016-17 | 19 | 622 |
| 2016-17 | 20 | 624 |
| 2016-17 | 21 | 631 |
| 2016-17 | 22 | 634 |
| 2016-17 | 23 | 640 |
| 2016-17 | 24 | 649 |
| 2016-17 | 25 | 650 |
| 2016-17 | 26 | 526 |
| 2016-17 | 27 | 652 |
| 2016-17 | 28 | 255 |
| 2016-17 | 29 | 654 |
| 2016-17 | 30 | 657 |
| 2016-17 | 31 | 661 |
| 2016-17 | 32 | 662 |
| 2016-17 | 33 | 663 |
| 2016-17 | 34 | 635 |
| 2016-17 | 35 | 668 |
| 2016-17 | 36 | 668 |
| 2016-17 | 37 | 672 |
| 2016-17 | 38 | 683 |
| 2017-18 | 1 | 511 |
| 2017-18 | 2 | 516 |
| 2017-18 | 3 | 531 |
| 2017-18 | 4 | 544 |
| 2017-18 | 5 | 546 |
| 2017-18 | 6 | 550 |
| 2017-18 | 7 | 554 |
| 2017-18 | 8 | 557 |
| 2017-18 | 9 | 559 |
| 2017-18 | 10 | 563 |
| 2017-18 | 11 | 566 |
| 2017-18 | 12 | 569 |
| 2017-18 | 13 | 570 |
| 2017-18 | 14 | 571 |
| 2017-18 | 15 | 574 |
| 2017-18 | 16 | 577 |
| 2017-18 | 17 | 578 |
| 2017-18 | 18 | 580 |
| 2017-18 | 19 | 585 |
| 2017-18 | 20 | 586 |
| 2017-18 | 21 | 526 |
| 2017-18 | 22 | 590 |
| 2017-18 | 23 | 598 |
| 2017-18 | 24 | 603 |
| 2017-18 | 25 | 613 |
| 2017-18 | 26 | 625 |
| 2017-18 | 27 | 627 |
| 2017-18 | 28 | 630 |
| 2017-18 | 29 | 633 |
| 2017-18 | 30 | 634 |
| 2017-18 | 31 | 267 |
| 2017-18 | 32 | 637 |
| 2017-18 | 33 | 641 |
| 2017-18 | 34 | 643 |
| 2017-18 | 35 | 407 |
| 2017-18 | 36 | 644 |
| 2017-18 | 37 | 645 |
| 2017-18 | 38 | 647 |
| 2018-19 | 1 | 525 |
| 2018-19 | 2 | 527 |
| 2018-19 | 3 | 531 |
| 2018-19 | 4 | 537 |
| 2018-19 | 5 | 537 |
| 2018-19 | 6 | 538 |
| 2018-19 | 7 | 539 |
| 2018-19 | 8 | 542 |
| 2018-19 | 9 | 546 |
| 2018-19 | 10 | 548 |
| 2018-19 | 11 | 550 |
| 2018-19 | 12 | 552 |
| 2018-19 | 13 | 553 |
| 2018-19 | 14 | 556 |
| 2018-19 | 15 | 558 |
| 2018-19 | 16 | 559 |
| 2018-19 | 17 | 562 |
| 2018-19 | 18 | 565 |
| 2018-19 | 19 | 568 |
| 2018-19 | 20 | 568 |
| 2018-19 | 21 | 570 |
| 2018-19 | 22 | 574 |
| 2018-19 | 23 | 578 |
| 2018-19 | 24 | 583 |
| 2018-19 | 25 | 597 |
| 2018-19 | 26 | 600 |
| 2018-19 | 27 | 484 |
| 2018-19 | 28 | 605 |
| 2018-19 | 29 | 606 |
| 2018-19 | 30 | 608 |
| 2018-19 | 31 | 306 |
| 2018-19 | 32 | 608 |
| 2018-19 | 33 | 373 |
| 2018-19 | 34 | 612 |
| 2018-19 | 35 | 614 |
| 2018-19 | 36 | 615 |
| 2018-19 | 37 | 616 |
| 2018-19 | 38 | 624 |
| 2019-20 | 1 | 526 |
| 2019-20 | 2 | 529 |
| 2019-20 | 3 | 529 |
| 2019-20 | 4 | 532 |
| 2019-20 | 5 | 539 |
| 2019-20 | 6 | 541 |
| 2019-20 | 7 | 546 |
| 2019-20 | 8 | 551 |
| 2019-20 | 9 | 555 |
| 2019-20 | 10 | 556 |
| 2019-20 | 11 | 557 |
| 2019-20 | 12 | 559 |
| 2019-20 | 13 | 560 |
| 2019-20 | 14 | 564 |
| 2019-20 | 15 | 565 |
| 2019-20 | 16 | 569 |
| 2019-20 | 17 | 576 |
| 2019-20 | 18 | 520 |
| 2019-20 | 19 | 580 |
| 2019-20 | 20 | 584 |
| 2019-20 | 21 | 586 |
| 2019-20 | 22 | 597 |
| 2019-20 | 23 | 606 |
| 2019-20 | 24 | 607 |
| 2019-20 | 25 | 619 |
| 2019-20 | 26 | 623 |
| 2019-20 | 27 | 624 |
| 2019-20 | 28 | 504 |
| 2019-20 | 29 | 628 |
| 2019-20 | 39 | 638 |
| 2019-20 | 40 | 644 |
| 2019-20 | 41 | 648 |
| 2019-20 | 42 | 652 |
| 2019-20 | 43 | 653 |
| 2019-20 | 44 | 654 |
| 2019-20 | 45 | 661 |
| 2019-20 | 46 | 665 |
| 2019-20 | 47 | 666 |
| 2020-21 | 1 | 425 |
| 2020-21 | 2 | 539 |
| 2020-21 | 3 | 554 |
| 2020-21 | 4 | 562 |
| 2020-21 | 5 | 583 |
| 2020-21 | 6 | 590 |
| 2020-21 | 7 | 597 |
| 2020-21 | 8 | 600 |
| 2020-21 | 9 | 601 |
| 2020-21 | 10 | 603 |
| 2020-21 | 11 | 541 |
| 2020-21 | 12 | 607 |
| 2020-21 | 13 | 607 |
| 2020-21 | 14 | 611 |
| 2020-21 | 15 | 614 |
| 2020-21 | 16 | 491 |
| 2020-21 | 17 | 560 |
| 2020-21 | 18 | 395 |
| 2020-21 | 19 | 610 |
| 2020-21 | 20 | 644 |
| 2020-21 | 21 | 650 |
| 2020-21 | 22 | 663 |
| 2020-21 | 23 | 667 |
| 2020-21 | 24 | 670 |
| 2020-21 | 25 | 673 |
| 2020-21 | 26 | 682 |
| 2020-21 | 27 | 685 |
| 2020-21 | 28 | 686 |
| 2020-21 | 29 | 277 |
| 2020-21 | 30 | 687 |
| 2020-21 | 31 | 691 |
| 2020-21 | 32 | 657 |
| 2020-21 | 33 | 555 |
| 2020-21 | 34 | 623 |
| 2020-21 | 35 | 703 |
| 2020-21 | 36 | 567 |
| 2020-21 | 37 | 706 |
| 2020-21 | 38 | 713 |
| 2021-22 | 1 | 554 |
| 2021-22 | 2 | 566 |
| 2021-22 | 3 | 577 |
| 2021-22 | 4 | 599 |
| 2021-22 | 5 | 606 |
| 2021-22 | 6 | 611 |
| 2021-22 | 7 | 613 |
| 2021-22 | 8 | 617 |
| 2021-22 | 9 | 618 |
| 2021-22 | 10 | 623 |
| 2021-22 | 11 | 624 |
| 2021-22 | 12 | 631 |
| 2021-22 | 13 | 574 |
| 2021-22 | 14 | 636 |
| 2021-22 | 15 | 640 |
| 2021-22 | 16 | 583 |
| 2021-22 | 17 | 460 |
| 2021-22 | 18 | 262 |
| 2021-22 | 19 | 450 |
| 2021-22 | 20 | 461 |
| 2021-22 | 21 | 618 |
| 2021-22 | 22 | 623 |
| 2021-22 | 23 | 697 |
| 2021-22 | 24 | 635 |
| 2021-22 | 25 | 633 |
| 2021-22 | 26 | 711 |
| 2021-22 | 27 | 602 |
| 2021-22 | 28 | 714 |
| 2021-22 | 29 | 714 |
| 2021-22 | 30 | 292 |
| 2021-22 | 31 | 720 |
| 2021-22 | 32 | 722 |
| 2021-22 | 33 | 612 |
| 2021-22 | 34 | 728 |
| 2021-22 | 35 | 730 |
| 2021-22 | 36 | 734 |
| 2021-22 | 37 | 703 |
| 2021-22 | 38 | 737 |
| 2022-23 | 1 | 573 |
| 2022-23 | 2 | 581 |
| 2022-23 | 3 | 592 |
| 2022-23 | 4 | 601 |
| 2022-23 | 5 | 608 |
| 2022-23 | 6 | 624 |
| 2022-23 | 8 | 440 |
| 2022-23 | 9 | 638 |
| 2022-23 | 10 | 641 |
| 2022-23 | 11 | 643 |
| 2022-23 | 12 | 585 |
| 2022-23 | 13 | 648 |
| 2022-23 | 14 | 656 |
| 2022-23 | 15 | 661 |
| 2022-23 | 16 | 667 |
| 2022-23 | 17 | 675 |
| 2022-23 | 18 | 678 |
| 2022-23 | 19 | 689 |
| 2022-23 | 20 | 699 |
| 2022-23 | 21 | 707 |
| 2022-23 | 22 | 740 |
| 2022-23 | 23 | 745 |
| 2022-23 | 24 | 746 |
| 2022-23 | 25 | 602 |
| 2022-23 | 26 | 747 |
| 2022-23 | 27 | 750 |
| 2022-23 | 28 | 539 |
| 2022-23 | 29 | 754 |
| 2022-23 | 30 | 754 |
| 2022-23 | 31 | 759 |
| 2022-23 | 32 | 611 |
| 2022-23 | 33 | 761 |
| 2022-23 | 34 | 762 |
| 2022-23 | 35 | 763 |
| 2022-23 | 36 | 765 |
| 2022-23 | 37 | 775 |
| 2022-23 | 38 | 778 |
| 2023-24 | 1 | 658 |
| 2023-24 | 2 | 595 |
| 2023-24 | 3 | 685 |
| 2023-24 | 4 | 703 |
| 2023-24 | 5 | 717 |
| 2023-24 | 6 | 718 |
| 2023-24 | 7 | 722 |
| 2023-24 | 8 | 725 |
| 2023-24 | 9 | 730 |
| 2023-24 | 10 | 731 |
| 2023-24 | 11 | 733 |
| 2023-24 | 12 | 743 |
| 2023-24 | 13 | 747 |
| 2023-24 | 14 | 753 |
| 2023-24 | 15 | 757 |
| 2023-24 | 16 | 758 |
| 2023-24 | 17 | 686 |
| 2023-24 | 18 | 700 |
| 2023-24 | 19 | 771 |
| 2023-24 | 20 | 773 |
| 2023-24 | 21 | 787 |
| 2023-24 | 22 | 799 |
| 2023-24 | 23 | 809 |
| 2023-24 | 24 | 814 |
| 2023-24 | 25 | 823 |
| 2023-24 | 26 | 649 |
| 2023-24 | 27 | 832 |
| 2023-24 | 28 | 838 |
| 2023-24 | 29 | 344 |
| 2023-24 | 30 | 842 |
| 2023-24 | 31 | 844 |
| 2023-24 | 32 | 846 |
| 2023-24 | 33 | 851 |
| 2023-24 | 34 | 815 |
| 2023-24 | 35 | 859 |
| 2023-24 | 36 | 859 |
| 2023-24 | 37 | 861 |
| 2023-24 | 38 | 865 |
| 2024-25 | 1 | 616 |
| 2024-25 | 2 | 627 |
| 2024-25 | 3 | 648 |
| 2024-25 | 4 | 659 |
| 2024-25 | 5 | 661 |
| 2024-25 | 6 | 664 |
| 2024-25 | 7 | 666 |
| 2024-25 | 8 | 667 |
| 2024-25 | 9 | 670 |
| 2024-25 | 10 | 674 |
| 2024-25 | 11 | 678 |
| 2024-25 | 12 | 684 |
| 2024-25 | 13 | 690 |
| 2024-25 | 14 | 693 |
| 2024-25 | 15 | 630 |
| 2024-25 | 16 | 701 |
| 2024-25 | 17 | 701 |
| 2024-25 | 18 | 705 |
| 2024-25 | 19 | 709 |
| 2024-25 | 20 | 711 |
| 2024-25 | 21 | 724 |
| 2024-25 | 22 | 729 |
| 2024-25 | 23 | 756 |
| 2024-25 | 24 | 762 |
| 2024-25 | 25 | 779 |
| 2024-25 | 26 | 783 |
| 2024-25 | 27 | 784 |
| 2024-25 | 28 | 789 |
| 2024-25 | 29 | 639 |
| 2024-25 | 30 | 791 |
| 2024-25 | 31 | 792 |
| 2024-25 | 32 | 798 |
| 2024-25 | 33 | 799 |
| 2024-25 | 34 | 645 |
| 2024-25 | 35 | 801 |
| 2024-25 | 36 | 801 |
| 2024-25 | 37 | 801 |
| 2024-25 | 38 | 804 |
| 2025-26 | 1 | 690 |
| 2025-26 | 2 | 705 |
| 2025-26 | 3 | 712 |
| 2025-26 | 4 | 740 |
| 2025-26 | 5 | 741 |
| 2025-26 | 6 | 742 |
| 2025-26 | 7 | 743 |
| 2025-26 | 8 | 745 |
| 2025-26 | 9 | 746 |
| 2025-26 | 10 | 747 |
| 2025-26 | 11 | 752 |
| 2025-26 | 12 | 755 |
| 2025-26 | 13 | 755 |
| 2025-26 | 14 | 758 |
| 2025-26 | 15 | 759 |
| 2025-26 | 16 | 760 |
| 2025-26 | 17 | 770 |
| 2025-26 | 18 | 775 |
| 2025-26 | 19 | 780 |
| 2025-26 | 20 | 790 |
| 2025-26 | 21 | 795 |
| 2025-26 | 22 | 799 |
| 2025-26 | 23 | 803 |
| 2025-26 | 24 | 811 |
| 2025-26 | 25 | 817 |
| 2025-26 | 26 | 817 |
| 2025-26 | 27 | 818 |
| 2025-26 | 28 | 819 |
| 2025-26 | 29 | 820 |
| 2025-26 | 30 | 822 |
| 2025-26 | 31 | 664 |
| 2025-26 | 32 | 826 |
| 2025-26 | 33 | 829 |
| 2025-26 | 34 | 582 |
| 2025-26 | 35 | 832 |
| 2025-26 | 36 | 838 |
| 2025-26 | 37 | 840 |
| 2025-26 | 38 | 841 |

## Feature Null Summary

| feature | null_count | null_rate |
| --- | --- | --- |
| prior_points_per_90 | 39648 | 0.162002 |
| prior_points_last1 | 2643 | 0.0107993 |
| prior_points_last5 | 2643 | 0.0107993 |
| prior_points_last3 | 2643 | 0.0107993 |
| prior_points_last10 | 2643 | 0.0107993 |
| prior_minutes_last3 | 2643 | 0.0107993 |
| prior_minutes_last10 | 2643 | 0.0107993 |
| prior_minutes_last5 | 2643 | 0.0107993 |
| prior_goals_last5 | 2643 | 0.0107993 |
| prior_starts_last5 | 2643 | 0.0107993 |
| prior_assists_last5 | 2643 | 0.0107993 |
| prior_appearances_last5 | 2643 | 0.0107993 |
| prior_xg_last5 | 2643 | 0.0107993 |
| prior_bonus_last5 | 2643 | 0.0107993 |
| prior_saves_last5 | 2643 | 0.0107993 |
| prior_clean_sheets_last5 | 2643 | 0.0107993 |
| prior_xa_last5 | 2643 | 0.0107993 |
| prior_points_season | 0 | 0 |
| prior_minutes_total | 0 | 0 |
| prior_gameweeks_played | 0 | 0 |

## Required Null Counts

| Column | Null count |
|--------|------------|
| canonical_player_key | 0 |
| fpl_code | 0 |
| season | 0 |
| gameweek | 0 |
| player_name | 0 |
| player_source_id | 0 |
| source_history_row_id | 0 |

## Manual Sample Rechecks

| canonical_player_key | season | gameweek | player_name | expected_prior_points_last3 | actual_prior_points_last3 | check_passed |
| --- | --- | --- | --- | --- | --- | --- |
| 100059 | 2016-17 | 6 | Alberto_Moreno | 0.333333 | 0.333333 | True |
| 101178 | 2016-17 | 6 | James_Ward-Prowse | 1.66667 | 1.66667 | True |
| 101184 | 2016-17 | 6 | Calum_Chambers | 0 | 0 | True |
| 101668 | 2016-17 | 6 | Jamie_Vardy | 6.66667 | 6.66667 | True |
| 102738 | 2016-17 | 6 | Giannelli_Imbula | 1.33333 | 1.33333 | True |

## Sample Feature Rows

| canonical_player_key | fpl_code | season | gameweek | player_name | target_total_points | prior_points_last3 | prior_minutes_last3 | prior_points_season | feature_history_end_season | feature_history_end_gameweek |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100059 | 100059 | 2016-17 | 1 | Alberto_Moreno | 0 |  |  | 0 |  |  |
| 101178 | 101178 | 2016-17 | 1 | James_Ward-Prowse | 1 |  |  | 0 |  |  |
| 101184 | 101184 | 2016-17 | 1 | Calum_Chambers | 6 |  |  | 0 |  |  |
| 101668 | 101668 | 2016-17 | 1 | Jamie_Vardy | 2 |  |  | 0 |  |  |
| 102738 | 102738 | 2016-17 | 1 | Giannelli_Imbula | 3 |  |  | 0 |  |  |
| 102884 | 102884 | 2016-17 | 1 | Paulo_Gazzaniga | 0 |  |  | 0 |  |  |
| 103025 | 103025 | 2016-17 | 1 | Riyad_Mahrez | 8 |  |  | 0 |  |  |
| 103100 | 103100 | 2016-17 | 1 | Jonathan_Williams | 0 |  |  | 0 |  |  |
| 10318 | 10318 | 2016-17 | 1 | Maarten_Stekelenburg | 3 |  |  | 0 |  |  |
| 103192 | 103192 | 2016-17 | 1 | Kurt_Zouma | 0 |  |  | 0 |  |  |

## Protected Tier 2 Counts

| Table | Before | After | Status |
|-------|--------|-------|--------|
| players | 841 | 841 | UNCHANGED |
| teams | 20 | 20 | UNCHANGED |
| fixtures | 380 | 380 | UNCHANGED |
| gameweeks | 38 | 38 | UNCHANGED |
| player_gameweek_history | 29747 | 29747 | UNCHANGED |
| player_gameweek_features | 29747 | 29747 | UNCHANGED |

## Model Artifact Status

`git status --short models/saved` output during final validation should remain empty. In-script model artifact status: not touched by this script.

## Confirmation

- No model training happened.
- No model evaluation happened.
- No tuning happened.
- No Streamlit changes happened.
- No production model artifact work happened.
- Only fpl_player_features_v3 was written.
