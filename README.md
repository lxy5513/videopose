## envrionment configture
I use torch1.0.1 in conda    
`conda env create -f env_info_file.yml`


---


## DEMO

  
  
`python demo.py`  

```
指定输入视频input video   --viz-video  
指定输入视频the keypoints of input_video  --input-npz   
指定输出视频名称 the name of output video --viz-output   
指定输出的帧数the frames number of output video  --viz-limit 
```

<br> 

--- 
## handle video by hrnet 
`updata at 2019-04-17`     
`python hrnet_video.py --viz-video /path/to/video.mp4`

add hrnet 2D keypoints detection module, to realize the end to end 3D reconstruction  
添加hrnet 2D关键点检测模块,实现更高精读的3D video 重构   [`hrnet`](https://github.com/lxy5513/hrnet)

---

## handle video by alphapose
`python wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`
or
`python wild_video.py --viz-output output.gif --viz-video /path/to/video.mp4 --viz-limit 180`

---

## handle video with every frame keypoints
`python wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4 --input-npz /path/to/input_name.npz`


<br> 

---


## 相关模型下载reletive model download  

其中hrnet依赖的pose model address: https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)
yolov3 model地址下载方法: `wget https://pjreddie.com/media/files/yolov3.weights`   

videopose model address: https://dl.fbaipublicfiles.com/video-pose-3d/cpn-pt-243.bin
