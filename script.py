import solution as sol

data = [ 'videos/video-' + str(i) + '.avi' for i in range (0, 10) ]

res = [[i, sol.get_result_from_video(video)] for i, video in enumerate(data)]

with open('test/sol.txt', 'w') as file:
    file.write('RA 90/2014 Aleksandra Velas\nfile\tsum\n')
    for r in res:
        file.write('video-'+str(r[0])+'.avi\t'+str(r[1])+'\n')