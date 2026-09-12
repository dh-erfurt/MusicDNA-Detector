def test_humming_keeps_notes_alive_through_voicing_dips() -> None:
    """The humming profile keeps sustained notes together through voicing dips."""
    from musicdna_detector import analysis_config_for_profile

    humming = analysis_config_for_profile("humming").segmentation
    assert humming.state_unvoiced_note_penalty == 0.2
    for profile in ("default", "singing"):
        other = analysis_config_for_profile(profile).segmentation
        assert other.state_unvoiced_note_penalty == 1.5
