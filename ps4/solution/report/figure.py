import os

paths = os.listdir("report/figures")

figStr = ""

print(paths)

for path in paths:
    if not "." in path:
        nPath = "figures/" + path
        files = os.listdir(nPath)
        for file in files:

            figStr = "\n" + figStr + r'\begin{figure}[H]' + "\n" + "\centerline{\includegraphics[width=\linewidth]{figures/" + path + "/" + file + "}}" + "\n" + "\caption{}" + "\n" + "\label{fig:" +file.replace(".png","") + "}" + "\n" + "\end{figure}" + "\n\n"


    else:
        figStr = "\n" + figStr + r'\begin{figure}[H]' + "\n" + "\centerline{\includegraphics[width=\linewidth]{figures/" + path + "}}" + "\n" + "\caption{}" + "\n" + "\label{fig:" +path.replace(".png","") + "}" + "\n" + "\end{figure}" + "\n\n"



        
print(figStr)

