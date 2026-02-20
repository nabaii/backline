import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from backend.store.analytics_store import AnalyticsStore
from backend.builders.match_analytics_builder import MatchAnalyticsBuilder
from backend.bet_type.double_chance.double_chance import DoubleChanceWorkspace
from backend.bet_type.over_under.over_under import OverUnderWorkspace
from backend.bet_type.one_x_two.one_x_two import OneXTwoWorkspace
from backend.filters.filters import (
    GoalsScored,
    GoalsConceded,
    TeamMomentumFilter,
    OpponentMomentumFilter,
    TeamXG,
    OpponentXG,
    LastNGames,
    HeadToHead,
    VenueFilter,
    XGDifferenceFilter,
)
from backend.chart.chart_spec import ChartSpec, AxisSpec
from backend.metrics.metric_spec import HitRateMetric, SampleSizeMetric


def test_chart_spec():
    """Test that ChartSpec is correctly configured for DoubleChance"""
    print("\n" + "="*60)
    print("TEST: ChartSpec Configuration")
    print("="*60)
    
    # Load raw data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    # Build MatchAnalytics objects for each unique match
    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    # Initialize store and ingest data
    store = AnalyticsStore()
    store.ingest(match_analytics_list)

    print(f"✓ Loaded {len(match_analytics_list)} matches into the store")

    # Initialize workspace
    dc_workspace = DoubleChanceWorkspace(store)
    
    # Test chart spec
    chart = dc_workspace.chart_spec
    
    assert isinstance(chart, ChartSpec), "chart_spec should be a ChartSpec instance"
    print(f"✓ chart_spec is a ChartSpec instance")
    
    assert chart.chart_type == "scatter", f"Expected chart_type 'scatter', got {chart.chart_type}"
    print(f"✓ chart_type is 'scatter'")
    
    assert chart.title == "Double Chance: Performance vs Opponents", f"Unexpected title: {chart.title}"
    print(f"✓ title: {chart.title}")
    
    assert chart.description is not None, "description should not be None"
    print(f"✓ description: {chart.description}")
    
    # Test X-axis
    assert isinstance(chart.x_axis, AxisSpec), "x_axis should be AxisSpec"
    assert chart.x_axis.name == "opponent_id", f"x_axis name should be 'opponent_id', got {chart.x_axis.name}"
    assert chart.x_axis.label == "Opponent", f"x_axis label should be 'Opponent', got {chart.x_axis.label}"
    assert chart.x_axis.data_column == "opponent_id", f"x_axis data_column should be 'opponent_id', got {chart.x_axis.data_column}"
    print(f"✓ x_axis configured correctly: name='{chart.x_axis.name}', label='{chart.x_axis.label}'")
    
    # Test Y-axis
    assert isinstance(chart.y_axis, AxisSpec), "y_axis should be AxisSpec"
    assert chart.y_axis.name == "double_chance_outcome", f"y_axis name should be 'double_chance_outcome', got {chart.y_axis.name}"
    assert chart.y_axis.label == "Win/Draw (1) vs Loss (0)", f"Unexpected y_axis label: {chart.y_axis.label}"
    assert chart.y_axis.data_column == "double_chance_outcome", f"y_axis data_column should be 'double_chance_outcome', got {chart.y_axis.data_column}"
    print(f"✓ y_axis configured correctly: name='{chart.y_axis.name}', label='{chart.y_axis.label}'")
    
    print("\n✅ All ChartSpec tests passed!")
    return store, dc_workspace


def test_available_metrics():
    """Test that available_metrics are correctly configured"""
    print("\n" + "="*60)
    print("TEST: Available Metrics Configuration")
    print("="*60)
    
    # Load data and create workspace
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    dc_workspace = DoubleChanceWorkspace(store)
    
    # Test available metrics
    metrics = dc_workspace.available_metrics
    
    assert isinstance(metrics, list), "available_metrics should return a list"
    print(f"✓ available_metrics returns a list")
    
    assert len(metrics) == 2, f"Expected 2 metrics, got {len(metrics)}"
    print(f"✓ Has 2 metrics")
    
    # Check HitRateMetric
    hit_rate_metric = metrics[0]
    assert isinstance(hit_rate_metric, HitRateMetric), "First metric should be HitRateMetric"
    assert hit_rate_metric.key == "double_chance_hit_rate", f"Unexpected key: {hit_rate_metric.key}"
    assert hit_rate_metric.name == "Hit Rate (Win/Draw)", f"Unexpected name: {hit_rate_metric.name}"
    assert hit_rate_metric.outcome_column == "double_chance_outcome", f"Unexpected outcome_column: {hit_rate_metric.outcome_column}"
    print(f"✓ HitRateMetric: key='{hit_rate_metric.key}', name='{hit_rate_metric.name}'")
    
    # Check SampleSizeMetric
    sample_size_metric = metrics[1]
    assert isinstance(sample_size_metric, SampleSizeMetric), "Second metric should be SampleSizeMetric"
    assert sample_size_metric.key == "sample_size", f"Unexpected key: {sample_size_metric.key}"
    assert sample_size_metric.name == "Sample Size", f"Unexpected name: {sample_size_metric.name}"
    print(f"✓ SampleSizeMetric: key='{sample_size_metric.key}', name='{sample_size_metric.name}'")
    
    print("\n✅ All Available Metrics tests passed!")


def test_metric_computation():
    """Test that metrics can be computed correctly on evidence"""
    print("\n" + "="*60)
    print("TEST: Metric Computation on Evidence")
    print("="*60)
    
    # Load data and create workspace
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    print(f"✓ Loaded {len(match_analytics_list)} matches into the store")

    # Initialize workspace
    dc_workspace = DoubleChanceWorkspace(store)

    # Target match and perspective
    target_match_id = 12436438
    perspective = "home"

    # Build filter: GoalsScored > 1
    filters = [
        GoalsScored.build(operator='>', value=1)
    ]

    # Query using the workspace API
    evidence = dc_workspace.get_evidence(
        match_id=target_match_id,
        bet_type="double_chance",
        filters=filters,
        perspective=perspective
    )

    print(f"\n✓ Retrieved evidence for match {target_match_id}")
    print(f"  Perspective: {perspective}")
    print(f"  Sample size: {evidence.sample_size}")
    print(f"  DataFrame shape: {evidence.df.shape}")
    print(f"  Columns: {list(evidence.df.columns)}")
    
    # Verify evidence has the required columns
    assert "double_chance_outcome" in evidence.df.columns, "Evidence should have 'double_chance_outcome' column"
    print(f"✓ Evidence has 'double_chance_outcome' column")
    
    assert "opponent_id" in evidence.df.columns, "Evidence should have 'opponent_id' column"
    print(f"✓ Evidence has 'opponent_id' column")
    
    # Get metrics from workspace
    metrics = dc_workspace.available_metrics
    hit_rate_metric = metrics[0]
    sample_size_metric = metrics[1]
    
    # Compute metrics
    hit_rate = hit_rate_metric.compute(evidence)
    sample_size = sample_size_metric.compute(evidence)
    
    print(f"\n✓ Computed metrics successfully")
    print(f"  Hit Rate: {hit_rate:.2%} ({int(hit_rate * sample_size)}/{sample_size})")
    print(f"  Sample Size: {sample_size}")
    
    # Verify metric values
    assert isinstance(hit_rate, float), f"Hit rate should be float, got {type(hit_rate)}"
    assert 0.0 <= hit_rate <= 1.0, f"Hit rate should be between 0 and 1, got {hit_rate}"
    print(f"✓ Hit rate is valid (0-1 range)")
    
    assert isinstance(sample_size, int), f"Sample size should be int, got {type(sample_size)}"
    assert sample_size >= 0, f"Sample size should be non-negative, got {sample_size}"
    print(f"✓ Sample size is valid")
    
    # Verify hit rate calculation
    expected_hits = (evidence.df["double_chance_outcome"] == 1).sum()
    expected_hit_rate = expected_hits / sample_size if sample_size > 0 else 0.0
    assert abs(hit_rate - expected_hit_rate) < 1e-6, f"Hit rate mismatch: computed={hit_rate}, expected={expected_hit_rate}"
    print(f"✓ Hit rate calculation verified")
    
    print("\n✅ All Metric Computation tests passed!")


def test_evidence_data_integrity():
    """Test that the evidence data is correctly enriched"""
    print("\n" + "="*60)
    print("TEST: Evidence Data Integrity")
    print("="*60)
    
    # Load data and create workspace
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    dc_workspace = DoubleChanceWorkspace(store)

    target_match_id = 12436438
    perspective = "home"
    filters = [GoalsScored.build(operator='>', value=1)]

    # Get evidence
    evidence = dc_workspace.get_evidence(
        match_id=target_match_id,
        bet_type="double_chance",
        filters=filters,
        perspective=perspective
    )

    df = evidence.df
    
    print(f"✓ Evidence retrieved with {len(df)} rows")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Verify outcome column values are 0 or 1
    outcome_values = df["double_chance_outcome"].unique()
    assert set(outcome_values).issubset({0, 1}), f"Outcome should only contain 0 or 1, got {outcome_values}"
    print(f"\n✓ double_chance_outcome contains only 0 or 1")
    
    # Verify opponent_id is present
    assert len(df["opponent_id"].unique()) > 0, "opponent_id should have values"
    print(f"✓ opponent_id is populated with {len(df['opponent_id'].unique())} unique opponents")
    
    # Verify evidence metadata
    assert evidence.perspective == perspective, f"Perspective mismatch"
    print(f"✓ Evidence perspective: {evidence.perspective}")
    
    assert evidence.bet_type == "double_chance", f"Bet type mismatch"
    print(f"✓ Evidence bet_type: {evidence.bet_type}")
    
    assert evidence.outcome_feature == "double_chance_outcome", f"Outcome feature mismatch"
    print(f"✓ Evidence outcome_feature: {evidence.outcome_feature}")
    
    print("\n✅ All Evidence Data Integrity tests passed!")


def test_all_filters():
    """Test all filter types"""
    print("\n" + "="*60)
    print("TEST: All Filters")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    dc_workspace = DoubleChanceWorkspace(store)

    target_match_id = 12436438
    perspective = "home"
    
    # Test GoalsScored filter
    print("\n• Testing GoalsScored filter...")
    filter_goals = GoalsScored.build(operator='>', value=1)
    assert filter_goals.key == 'goals_scored'
    assert filter_goals.field == 'goals_scored'
    assert filter_goals.operator == '>'
    assert filter_goals.value == 1
    print("  ✓ GoalsScored filter created successfully")
    
    # Test GoalsConceded filter
    print("• Testing GoalsConceded filter...")
    filter_conceded = GoalsConceded.build(operator='<', value=2)
    assert filter_conceded.key == 'goals_conceded'
    assert filter_conceded.field == 'opponent_goals'
    assert filter_conceded.operator == '<'
    assert filter_conceded.value == 2
    print("  ✓ GoalsConceded filter created successfully")
    
    # Test TeamMomentumFilter
    print("• Testing TeamMomentumFilter...")
    filter_team_mom = TeamMomentumFilter.build(operator='between', value=(3.0, 6.5), perspective='home')
    assert filter_team_mom.key == 'team_momentum'
    assert filter_team_mom.field == 'home_momentum'  # canonical field; store applies venue-aware logic
    assert filter_team_mom.operator == 'between'
    assert filter_team_mom.value == (3.0, 6.5)
    print("  ✓ TeamMomentumFilter (home) created successfully")

    filter_team_mom_away = TeamMomentumFilter.build(operator='between', value=(7.0, 4.0), perspective='away')
    assert filter_team_mom_away.field == 'home_momentum'
    assert filter_team_mom_away.value == (4.0, 7.0)  # min/max normalized
    print("  ✓ TeamMomentumFilter (away) created successfully")

    # Test OpponentMomentumFilter
    print("• Testing OpponentMomentumFilter...")
    filter_opp_mom = OpponentMomentumFilter.build(operator='between', value=(1.0, 3.0), perspective='home')
    assert filter_opp_mom.key == 'opponent_momentum'
    assert filter_opp_mom.field == 'away_momentum'  # canonical field; store applies venue-aware logic
    assert filter_opp_mom.operator == 'between'
    assert filter_opp_mom.value == (1.0, 3.0)
    print("  ✓ OpponentMomentumFilter (home perspective) created successfully")
    # Test TeamXG filter
    print("• Testing TeamXG filter...")
    filter_team_xg = TeamXG.build(operator='>=', value=1.5)
    assert filter_team_xg.key == 'team_xg'
    assert filter_team_xg.field == 'team_xg'
    assert filter_team_xg.operator == '>='
    assert filter_team_xg.value == 1.5
    print("  ✓ TeamXG filter created successfully")
    
    # Test OpponentXG filter
    print("• Testing OpponentXG filter...")
    filter_opp_xg = OpponentXG.build(operator='<', value=1.0)
    assert filter_opp_xg.key == 'opponent_xg'
    assert filter_opp_xg.field == 'opponent_xg'
    assert filter_opp_xg.operator == '<'
    assert filter_opp_xg.value == 1.0
    print("  ✓ OpponentXG filter created successfully")
    
    # Test LastNGames filter
    print("• Testing LastNGames filter...")
    filter_last_n = LastNGames.build(operator='<=', value=10)
    assert filter_last_n.key == 'last_n_games'
    assert filter_last_n.operator == '<='
    assert filter_last_n.value == 10
    print("  ✓ LastNGames filter created successfully")
    
    # Test HeadToHead filter
    print("• Testing HeadToHead filter...")
    filter_h2h = HeadToHead.build(operator='==', value=True)
    assert filter_h2h.key == 'head_to_head'
    assert filter_h2h.value == True
    print("  ✓ HeadToHead filter created successfully")
    
    # Test VenueFilter
    print("• Testing VenueFilter...")
    filter_venue = VenueFilter.build(operator='==', value='home')
    assert filter_venue.key == 'venue'
    assert filter_venue.field == 'venue'
    assert filter_venue.value == 'home'
    print("  ✓ VenueFilter created successfully")
    
    # Test XGDifferenceFilter
    print("• Testing XGDifferenceFilter...")
    filter_xg_diff = XGDifferenceFilter.build(operator='>=', value=0.5)
    assert filter_xg_diff.key == 'xg_difference'
    assert filter_xg_diff.field == 'xg_diff'
    assert filter_xg_diff.operator == '>='
    assert filter_xg_diff.value == 0.5
    print("  ✓ XGDifferenceFilter created successfully")
    
    print("\n✅ All Filter tests passed!")


def test_over_under_workspace():
    """Test OverUnderWorkspace"""
    print("\n" + "="*60)
    print("TEST: OverUnderWorkspace")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    print(f"✓ Loaded {len(match_analytics_list)} matches into the store")

    # Initialize OverUnder workspace
    ou_workspace = OverUnderWorkspace(store)
    
    # Test basic properties
    assert ou_workspace.name == "over_under"
    print(f"✓ Workspace name: {ou_workspace.name}")
    
    # Test allowed filters
    allowed_filters = ou_workspace.allowed_filters
    assert len(allowed_filters) > 0
    print(f"✓ Has {len(allowed_filters)} allowed filters")
    
    # Test available metrics
    metrics = ou_workspace.available_metrics
    assert len(metrics) == 2
    print(f"✓ Has 2 available metrics")
    
    # Test chart spec
    chart = ou_workspace.chart_spec
    assert chart.chart_type == "scatter"
    assert "Total Goals" in chart.title
    print(f"✓ Chart spec configured: {chart.title}")
    
    # Test get_evidence with default line (2.5)
    target_match_id = 12436438
    perspective = "home"
    filters = [GoalsScored.build(operator='>', value=0)]
    
    evidence = ou_workspace.get_evidence(
        match_id=target_match_id,
        bet_type="over_under",
        filters=filters,
        perspective=perspective
    )
    
    assert "over_under_outcome" in evidence.df.columns
    assert "total_goals" in evidence.df.columns
    print(f"✓ Evidence retrieved with outcome and total_goals columns")
    
    # Test get_evidence with custom line
    evidence_custom = ou_workspace.get_evidence(
        match_id=target_match_id,
        bet_type="over_under",
        filters=filters,
        perspective=perspective,
        line=3.0
    )
    
    assert evidence_custom.sample_size > 0
    print(f"✓ Evidence retrieved with custom line (3.0): {evidence_custom.sample_size} records")
    
    print("\n✅ All OverUnderWorkspace tests passed!")


def test_one_x_two_workspace():
    """Test OneXTwoWorkspace for all outcome types (1, X, 2)"""
    print("\n" + "="*60)
    print("TEST: OneXTwoWorkspace")
    print("="*60)
    
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "season_df_v1.csv"
    raw_df = pd.read_csv(data_path)

    builder = MatchAnalyticsBuilder()
    match_analytics_list = []

    for match_id in raw_df["match_id"].unique():
        match_df = raw_df[raw_df["match_id"] == match_id].copy()
        match_analytics = builder.build(match_df)
        match_analytics_list.append(match_analytics)

    store = AnalyticsStore()
    store.ingest(match_analytics_list)
    print(f"✓ Loaded {len(match_analytics_list)} matches into the store")

    target_match_id = 12436438
    perspective = "home"
    filters = [GoalsScored.build(operator='>', value=0)]
    
    # Test Win (1)
    print("\n• Testing Win outcome (1)...")
    ws_win = OneXTwoWorkspace(store, outcome_type="1")
    assert ws_win.outcome_type == "1"
    assert ws_win.name == "one_x_two"
    print("  ✓ Win workspace created")
    
    evidence_win = ws_win.get_evidence(
        match_id=target_match_id,
        bet_type="one_x_two",
        filters=filters,
        perspective=perspective
    )
    assert "one_x_two_outcome" in evidence_win.df.columns
    assert "one_x_two_result" in evidence_win.df.columns
    print(f"  ✓ Evidence retrieved for Win: {evidence_win.sample_size} records")
    
    metrics_win = ws_win.available_metrics
    assert len(metrics_win) == 2
    assert "Win" in metrics_win[0].name
    print(f"  ✓ Metrics configured for Win: {metrics_win[0].name}")
    
    # Test Draw (X)
    print("\n• Testing Draw outcome (X)...")
    ws_draw = OneXTwoWorkspace(store, outcome_type="X")
    assert ws_draw.outcome_type == "X"
    print("  ✓ Draw workspace created")
    
    evidence_draw = ws_draw.get_evidence(
        match_id=target_match_id,
        bet_type="one_x_two",
        filters=filters,
        perspective=perspective
    )
    assert "one_x_two_outcome" in evidence_draw.df.columns
    print(f"  ✓ Evidence retrieved for Draw: {evidence_draw.sample_size} records")
    
    metrics_draw = ws_draw.available_metrics
    assert "Draw" in metrics_draw[0].name
    print(f"  ✓ Metrics configured for Draw: {metrics_draw[0].name}")
    
    # Test Loss (2)
    print("\n• Testing Loss outcome (2)...")
    ws_loss = OneXTwoWorkspace(store, outcome_type="2")
    assert ws_loss.outcome_type == "2"
    print("  ✓ Loss workspace created")
    
    evidence_loss = ws_loss.get_evidence(
        match_id=target_match_id,
        bet_type="one_x_two",
        filters=filters,
        perspective=perspective
    )
    assert "one_x_two_outcome" in evidence_loss.df.columns
    print(f"  ✓ Evidence retrieved for Loss: {evidence_loss.sample_size} records")
    
    metrics_loss = ws_loss.available_metrics
    assert "Loss" in metrics_loss[0].name
    print(f"  ✓ Metrics configured for Loss: {metrics_loss[0].name}")
    
    # Verify outcome logic
    print("\n• Testing outcome logic...")
    df_win = evidence_win.df
    df_draw = evidence_draw.df
    df_loss = evidence_loss.df
    
    # For a given row, outcomes should be mutually exclusive
    # i.e., at most one of outcome_win, outcome_draw, outcome_loss should be 1
    for idx in range(min(5, len(df_win))):
        result_win = df_win.iloc[idx]["one_x_two_result"]
        result_draw = df_draw.iloc[idx]["one_x_two_result"]
        result_loss = df_loss.iloc[idx]["one_x_two_result"]
        
        # All should be the same result value
        assert result_win == result_draw == result_loss
    
    print("  ✓ Outcome logic verified (results are consistent)")
    
    # Test chart spec
    print("\n• Testing chart spec...")
    chart = ws_win.chart_spec
    assert chart.chart_type == "scatter"
    assert "Win" in chart.title
    print(f"  ✓ Chart spec configured: {chart.title}")
    
    chart_draw = ws_draw.chart_spec
    assert "Draw" in chart_draw.title
    print(f"  ✓ Draw chart spec: {chart_draw.title}")
    
    chart_loss = ws_loss.chart_spec
    assert "Loss" in chart_loss.title
    print(f"  ✓ Loss chart spec: {chart_loss.title}")
    
    print("\n✅ All OneXTwoWorkspace tests passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BACKLINE COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    try:
        # Double Chance tests
        print("\n" + "█"*60)
        print("SECTION 1: DOUBLE CHANCE WORKSPACE")
        print("█"*60)
        store, workspace = test_chart_spec()
        test_available_metrics()
        test_metric_computation()
        test_evidence_data_integrity()
        
        # Filter tests
        print("\n" + "█"*60)
        print("SECTION 2: FILTERS")
        print("█"*60)
        test_all_filters()
        
        # Over/Under tests
        print("\n" + "█"*60)
        print("SECTION 3: OVER/UNDER WORKSPACE")
        print("█"*60)
        test_over_under_workspace()
        
        # 1x2 tests
        print("\n" + "█"*60)
        print("SECTION 4: 1X2 WORKSPACE")
        print("█"*60)
        test_one_x_two_workspace()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

