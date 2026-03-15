from voxatlas.cli.main import build_parser


def test_features_list_command_shows_registered_features(capsys):
    parser = build_parser()
    args = parser.parse_args(["features", "list", "--no-color"])

    args.func(args)

    captured = capsys.readouterr()

    assert "feature_name" in captured.out
    assert "status" in captured.out
    assert "lexical.frequency.word" in captured.out


def test_features_list_command_shows_missing_dependency_entries(capsys):
    parser = build_parser()
    args = parser.parse_args(["features", "list", "--no-color"])

    args.func(args)

    captured = capsys.readouterr()

    assert "acoustic.envelope.hilbert" in captured.out
    assert "missing:scipy" in captured.out
