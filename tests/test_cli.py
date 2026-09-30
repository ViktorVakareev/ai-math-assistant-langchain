from math_assistant.cli import main


def test_cli_answer(capsys):
    assert main(["What is 25 divided by 4?", "--provider", "offline"]) == 0
    assert "The result is 6.25." in capsys.readouterr().out


def test_cli_trace_shows_the_tool_loop(capsys):
    main(["Add 25 and 15, then multiply by 2", "-p", "offline", "--trace"])
    out = capsys.readouterr().out
    for expected in ["Human", "add_numbers", '{"result": 40.0}', "multiply_numbers", "The result is 80."]:
        assert expected in out
