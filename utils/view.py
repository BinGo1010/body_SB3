import mujoco
import mujoco.viewer
import os

# 加载你的 XML 文件
# 因为脚本和 xml 文件在同一个目录下，所以直接写文件名即可
xml_path = '/home/lvchen/miniconda3/envs/myosuite/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml'
# xml_path = "/home/lvchen/body_SB3_11-7/resources/exo/hip_exo.xml"

if not os.path.exists(xml_path):
    print(f"错误: 找不到文件 '{xml_path}'")
else:
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)

        # 启动可视化界面
        print(f"正在为 '{xml_path}' 启动 MuJoCo Viewer...")
        mujoco.viewer.launch(model, data)
    except Exception as e:
        print(f"加载或启动模型时出错: {e}") #<!--外骨骼侧展自由度限制-->
