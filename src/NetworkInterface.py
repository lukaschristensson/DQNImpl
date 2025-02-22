import socket
import threading
import DQN
import json


def argmax(lst, filtr=None):
    max_val = float('-inf')
    max_ind = 0
    for i, v in enumerate(lst):
        if v > max_val and (not filtr or i in filtr):
            max_ind = i
            max_val = v
    return max_ind

def require(js, *args):
    if any(a not in js for a in args):
        raise Exception(f'Requirement not met:{args} in\n {json.dumps(js, indent=4)}')
class NetworkInterface:
    def __init__(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(32)
            self.DQNs = {}
            while 1:
                print('Waiting for new connection...')
                ca_tuple = s.accept()
                t = threading.Thread(target=lambda: self.handle_conn(*ca_tuple))
                t.start()

    def handle_conn(self, conn, addr):
        try:
            print(f'connected to: {addr}')
            while True:
                data = conn.recv(4096).decode()
                try:
                    jsn = json.loads(data)
                except:
                    print(f'invalid json: {data}')
                    conn.close()
                    return

                require(jsn, 'fun')
                if jsn['fun'] == 'new_agent':
                    require(jsn, 'name', 'state_dim', 'action_dim')

                    self.DQNs[jsn['name']] = DQN.DQN(jsn['state_dim'], jsn['action_dim'])
                elif jsn['fun'] == 'get_action':
                    require(jsn, 'name', 'state')

                    dqn = self.DQNs[jsn['name']]
                    state = jsn['state']
                    action = argmax(dqn.action(state), jsn['action_filter'] if 'action_filter' in jsn else None)
                    s = f'{action}\n' # have to add a line break for Lua socket to recognize the end of the message
                    conn.send(s.encode())
                elif jsn['fun'] == 'store_in_replay':
                    require(jsn, 'name', 'state', 'action', 'reward', 'next_state', 'done')

                    self.DQNs[jsn['name']].store_in_replay(
                        jsn['state'],
                        jsn['action'],
                        jsn['reward'],
                        jsn['next_state'],
                        jsn['done']
                    )
                    self.DQNs[jsn['name']].replay()
        except Exception as e:
            conn.close()
            raise e



NetworkInterface("127.0.0.1", 100)