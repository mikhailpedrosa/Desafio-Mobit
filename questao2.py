import subprocess
import os

path = "C:/Users/Mikhail Pedrosa/PycharmProjects/mobit/questao2/darknet-master/build/darknet/x64"
os.chdir(path)

cmd = "./darknet_no_gpu.exe detect cfg/yolov3.cfg yolov3.weights data/pessoas.jpg"

output = subprocess.check_output(cmd.split())
output = output.decode("utf-8").split("\n")

numPeople = len([i.split(":")[0] for i in output if i.split(":")[0] == 'person'])

print(output[0])
print("Na imagem detectamos {0} pessoas.".format(numPeople))
