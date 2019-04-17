## 环境配置  
我是用的是torch1.0.1 版本 conda    
`conda env create -f env_info_file.yml`


---


## DEMO

  
  
`python demo.py`  

```
指定输入视频  --viz-video  
指定输入视频的关节点  --input-npz   
指定输出视频名称  --viz-output   
```

<br> 

--- 
## handle video by hrnet 
`updata at 2019-04-17`     
`python hrnet_video.py --viz-video /path/to/video.mp4`

添加hrnet 2D关键点检测模块,实现更高精读的3D video 重构   [`hrnet`](https://github.com/lxy5513/hrnet)

---

## handle video by alphapose
`python wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4`
or
`python wild_video.py --viz-output output.gif --viz-video /path/to/video.mp4 --viz-limit 180`

---

## handle video with every frame keypoints
`python wild_video.py --viz-output output.mp4 --viz-video /path/to/video.mp4 --input-npz /path/to/input_name.npz`


---

## 相关模型下载

其中hrnet依赖的pose模型地址https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA)
yolo模型地址下载方法`wget https://pjreddie.com/media/files/yolov3.weights`   

videopose的模型地址https://dl.fbaipublicfiles.com/video-pose-3d/cpn-pt-243.bin
