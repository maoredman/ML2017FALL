# -!- coding: utf-8 -!-
import socket
import pickle
def main():
    TCP_IP = 'cl5.learner.csie.ntu.edu.tw'
    TCP_PORT = 9487
    BUFFER_SIZE = 50000
    # ["question", "passage"]
    MESSAGE = ['亞洲帕拉運動會在哪裡舉行?', '2010年亞洲帕拉運動會又稱2010年廣州亞殘運會，於2010年12月12至19日在中國廣東省廣州市舉行，是一場由存在身體殘疾的運動員參與的綜合運動會。亞洲帕拉運動會的前身是遠東及南太平洋殘疾人運動會，本屆廣州亞殘會是首屆亞洲帕拉運動會，於2010年亞洲運動會結束兩周後舉行。']
    # dump as string
    MESSAGE = pickle.dumps(MESSAGE)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    s.send(MESSAGE)
    # get pickle of answer text
    data = s.recv(BUFFER_SIZE)
    # server will send a tuple of ("<answer_text>", "start position attention score", "end position attention score")
    print(pickle.loads(data))
    s.close()

if __name__ == '__main__':
    main()