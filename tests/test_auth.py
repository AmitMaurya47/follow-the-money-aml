from aml_copilot.auth import DEMO_PASSWORD_HINT, DEMO_USER, check_login


def test_demo_login_accepted_and_case_insensitive_username():
    assert check_login(DEMO_USER, DEMO_PASSWORD_HINT)
    assert check_login("  DEMO ", DEMO_PASSWORD_HINT)


def test_wrong_or_empty_credentials_rejected():
    assert not check_login(DEMO_USER, "wrong")
    assert not check_login("someone", DEMO_PASSWORD_HINT)
    assert not check_login("", "")
    assert not check_login(DEMO_USER, "")
