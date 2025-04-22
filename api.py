import http.client
import ast

urls = ["/aip2pgaming/api/rl/gw.php", 
        "/aip2pgaming/api/rl/score.php",
        "/aip2pgaming/api/index.php",
        "/aip2pgaming/api/rl/reset.php"]

id = '3672'
key = '6f7504789dda4c53374b'
teamId = '1459'

headers = {
  'userId': id,
  'x-api-key': key,
  'Content-Type': 'application/x-www-form-urlencoded',
  'Cookie': 'humans_21909=1'
}

def make_post_request(parameters: str, url: str) -> dict:
    conn = http.client.HTTPSConnection("www.notexponential.com")
    conn.request("POST", url, parameters, headers)
    response = conn.getresponse()
    data = response.read().decode()
    return ast.literal_eval(data)

def make_get_request(parameters: str, url: str) -> dict:
    conn = http.client.HTTPSConnection("www.notexponential.com")
    full_path = url + "?" + parameters
    conn.request("GET", full_path, None, headers)
    response = conn.getresponse()
    data = response.read().decode()
    return ast.literal_eval(data)


def get_runs(teamId: str, count: int) -> dict:
    payload = f"type=runs&teamId={teamId}&count={count}"
    res = make_get_request(payload, urls[1])
    print(res)
    assert res['code'] == "OK", 'get_runs'
    return res


def get_location(teamId: str) -> dict:
    payload = f"type=location&teamId={teamId}"
    res = make_get_request(payload, urls[0])
    print(res)
    assert res['code'] == "OK", 'get_location'
    return res["world"], res["state"]


def enter_world(teamId: str, worldId: str) -> dict:
    payload = f"type=enter&teamId={teamId}&worldId={worldId}"
    res = make_post_request(payload, urls[0])
    print(res)
    assert res['code'] == "OK", 'enter_world'
    return res['state']

def make_move(teamId: str, move: str, worldId: int) -> dict:
    payload = f"type=move&teamId={teamId}&move={move}&worldId={worldId}"
    res = make_post_request(payload, urls[0])
    print(res)
    assert res['code'] == "OK", "make_move"
    return res["reward"], res["newState"]

def get_score(teamId: str) -> dict:
    payload = f"type=score&teamId={teamId}"
    res = make_get_request(payload, urls[1])
    print(res)
    assert res['code'] == "OK", 'get_score'
    return res


def reset_team(teamId: str, otp: str):
    payload = f"teamId={teamId}&otp={otp}"
    res = make_get_request(payload, urls[3])
    print(res)
    assert res['code'] == "OK", "reset_team"
    return res


def create_team(tname: str) -> dict:
    payload = f"type=team&name={tname}"
    res = make_post_request(payload, urls[2])
    print(res)
    assert res['code'] == 'OK', 'create_team'
    return res['teamId']
        

def add_team_member(teamId: str, userId: str)-> dict:
    payload = f"type=member&userId={userId}&teamId={teamId}"
    res = make_post_request(payload, urls[2])
    print(res)
    assert res['code'] == 'OK', 'add_team_member'
    return res


def remove_team_member(teamId: str, userId: str) -> dict:
    payload = f"type=removeMember&userId={userId}&teamId={teamId}"
    res = make_post_request(payload, urls[2])
    print(res)
    assert res['code'] == 'OK', 'remove_team_member'
    return res


def get_team_members(teamId: str) -> dict:
    payload = f"type=team&teamId={teamId}"
    res = make_get_request(payload, urls[2])
    print(res)
    assert res['code'] == 'OK', 'get_team_members'
    return res


def get_my_team() -> dict:
    payload = f"type=myTeams"
    res = make_get_request(payload, urls[2])
    print(res)
    assert res['code'] == 'OK', 'get_my_team'
    return res 


# create_team("TESTRF")
# add_team_member(teamId, id)
# get_location(teamId)
# get_runs(teamId, 1)
# reset_team(teamId, "5712768807")
# get_location(teamId)
# make_move(teamId, "S", 0)
# enter_world(teamId, "1")
# make_move(teamId, "S", "1")