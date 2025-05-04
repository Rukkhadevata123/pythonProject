# 3D模型渲染器批处理系统 - 支持并行任务处理
# 功能：渲染OBJ模型，支持纹理和深度图生成，支持相机动画和光源动画

# ===== 基本渲染参数配置 =====
OBJ_FILE		?= bunny/bunny2k_f.obj
MODEL_NAME	  ?= $(basename $(notdir $(OBJ_FILE)))
WIDTH		   ?= 1200
HEIGHT		  ?= 1200
PROJECTION	  ?= orthographic
FRAMES		  ?= 120
OUTPUT_DIR	  ?= output_$(MODEL_NAME)
FPS			 ?= 30
COLOR_VIDEO	 ?= $(MODEL_NAME)_$(PROJECTION).mp4
DEPTH_VIDEO	 ?= $(MODEL_NAME)_$(PROJECTION)_depth.mp4

# ===== 渲染选项 =====
TEXTURE		 ?=
TEXTURE_TYPE	?= checkerboard
TEXTURE_SIZE	?= 512
USE_TEXTURE	 ?= 0
USE_MATERIALS   ?= 1
COLORIZE		?= 1
USE_LIGHTING	?= 1	   # 是否使用光照
LIGHT_MODEL	 ?= phong   # 光照模型类型: phong, blinn-phong
AMBIENT		 ?= 0.2	 # 环境光强度
DIFFUSE		 ?= 0.6	 # 漫反射强度
SPECULAR		?= 0.2	 # 高光强度
SHININESS	   ?= 32.0	# 高光锐度
RENDER_DEPTH	?= 1
DEPTH_MIN	   ?= 1
DEPTH_MAX	   ?= 99
ZBUFFER		 ?= 1
FOCAL		   ?= 2.0

# ===== 光源参数 =====
LIGHT_TYPE	  ?= point   # 光源类型: directional 或 point
# 方向光参数
LIGHT_DIR	   ?= 1,-1,1  # 方向光方向
# 点光源参数
LIGHT_POS	   ?= 0,0,1   # 点光源初始位置
LIGHT_ATTEN	 ?= 1.0,0.09,0.032  # 点光源衰减系数 (常量,线性,平方)
# 光源动画参数
LIGHT_ANIMATION ?= vertical # 光源动画: none, vertical, horizontal, circular, pulse, figure8, spiral, custom
LIGHT_RANGE	 ?= 1.0	  # 光源移动范围/幅度
# 自定义光源动画表达式 (t=0..1)
CUSTOM_X_EXPR   ?= sin(2*pi*t)
CUSTOM_Y_EXPR   ?= cos(2*pi*t)
CUSTOM_Z_EXPR   ?= 0

# ===== 相机参数 (新增) =====
CAMERA_TYPE	 ?= orbit	# 相机动画类型: yaw, pitch, roll, orbit
CAMERA_FROM	 ?= 0,0,4	# 相机位置
CAMERA_AT	   ?= 0,0,0	# 观察点
CAMERA_UP	   ?= 0,1,0	# 上方向向量
CAMERA_FOV	  ?= 45.0	 # 视场角(度)

# ===== 动画类型 =====
# model: 旋转模型, angle 参数生效
# light: 移动光源, light-frame 参数生效, 通常禁用 colorize 和 depth
# camera: 相机动画, camera_type 和 frame 参数生效
ANIMATION_TYPE  ?= model

# ===== 并行处理配置 =====
# 获取CPU核心数，如果失败则默认为4
NPROC := $(shell nproc 2>/dev/null || echo 4)
# 将并行参数添加到 MAKEFLAGS 中，使子make调用也能并行
MAKEFLAGS += -j$(NPROC)

# ===== 帧序列生成 =====
FRAME_NUMS := $(shell seq 0 $(shell expr $(FRAMES) - 1))
# 更简洁地定义输出文件列表，不需要printf格式化，直接使用数字
COLOR_FRAME_FILES := $(patsubst %,$(OUTPUT_DIR)/color/frame_%.png,$(FRAME_NUMS))
DEPTH_FRAME_FILES := $(patsubst %,$(OUTPUT_DIR)/depth/frame_%.png,$(FRAME_NUMS))

# ===== 渲染选项构建 =====
# 基础选项
RENDER_OPTIONS = --obj $(OBJ_FILE) \
	--width $(WIDTH) --height $(HEIGHT) \
	--projection $(PROJECTION) \
	--focal $(FOCAL) \
	--output-dir $(OUTPUT_DIR) \
	--texture-type $(TEXTURE_TYPE) \
	--texture-size $(TEXTURE_SIZE) \
	--depth-min $(DEPTH_MIN) --depth-max $(DEPTH_MAX) \
	--total-frames $(FRAMES) \
	--animation-type $(ANIMATION_TYPE)

# 相机相关选项 (新增)
RENDER_OPTIONS += \
	--camera-type $(CAMERA_TYPE) \
	--camera-from "$(CAMERA_FROM)" \
	--camera-at "$(CAMERA_AT)" \
	--camera-up "$(CAMERA_UP)" \
	--camera-fov $(CAMERA_FOV)

# 光照相关选项 (如果启用)
ifneq ($(USE_LIGHTING), 0)
	RENDER_OPTIONS += \
		--light-model $(LIGHT_MODEL) \
		--ambient $(AMBIENT) \
		--diffuse $(DIFFUSE) \
		--specular $(SPECULAR) \
		--shininess $(SHININESS) \
		--light-type $(LIGHT_TYPE) \
		--light-dir "$(LIGHT_DIR)" \
		--light-pos "$(LIGHT_POS)" \
		--light-atten "$(LIGHT_ATTEN)" \
		--light-animation $(LIGHT_ANIMATION) \
		--light-range $(LIGHT_RANGE) \
		--custom-x-expr "$(CUSTOM_X_EXPR)" \
		--custom-y-expr "$(CUSTOM_Y_EXPR)" \
		--custom-z-expr "$(CUSTOM_Z_EXPR)"
else
	RENDER_OPTIONS += --no-lighting
endif

# 纹理相关选项
ifeq ($(USE_TEXTURE), 0)
	RENDER_OPTIONS += --no-texture
else
	ifneq ($(TEXTURE),)
		RENDER_OPTIONS += --texture $(TEXTURE)
	endif
endif

# 其他布尔选项
ifeq ($(USE_MATERIALS), 0)
	RENDER_OPTIONS += --no-materials
endif
ifeq ($(ZBUFFER), 0)
	RENDER_OPTIONS += --no-zbuffer
endif
ifeq ($(RENDER_DEPTH), 0)
	RENDER_OPTIONS += --no-depth
endif
ifeq ($(COLORIZE), 1)
	RENDER_OPTIONS += --colorize
endif

# 光源动画类型时的特殊处理
ifeq ($(ANIMATION_TYPE),light)
  override COLORIZE = 0
  override RENDER_DEPTH = 0
endif

# ===== 主要目标 =====
.PHONY: all color_only depth_only frames color_video depth_video render_single clean clean_videos help
.PHONY: light_vertical light_horizontal light_circular light_pulse light_figure8 light_spiral light_custom light_custom_fixed
.PHONY: camera_yaw camera_pitch camera_roll camera_orbit camera_all

all: color_video depth_video

color_only: color_video
depth_only: depth_video

# ===== 目录创建目标 =====
# 使用 Order-only prerequisite，确保目录存在但不触发规则重新执行
$(OUTPUT_DIR)/color $(OUTPUT_DIR)/depth:
	@mkdir -p $@

# ===== 单帧渲染规则 =====
# 直接在规则内部创建目录，避免每个帧都要检查目录依赖
# 修改计算角度的部分，避免使用科学记数法，并使用格式化输出
$(OUTPUT_DIR)/color/frame_%.png:
	@mkdir -p $(OUTPUT_DIR)/color $(OUTPUT_DIR)/depth
	@echo "渲染帧 $* ($(ANIMATION_TYPE) 动画)..."
	@frame_num=$*; \
	output_name="frame_$$frame_num"; \
	angle=0; \
	light_frame_arg=""; \
	if [ "$(ANIMATION_TYPE)" = "model" ]; then \
		if [ $(FRAMES) -gt 1 ]; then \
			angle=$$(echo "scale=4; $$frame_num * 360 / ($(FRAMES) - 1)" | bc); \
		else \
			angle=0; \
		fi; \
		echo "  - 角度: $$angle°"; \
		python3 main.py $(RENDER_OPTIONS) --angle "$$angle" --output "$$output_name"; \
	elif [ "$(ANIMATION_TYPE)" = "camera" ]; then \
		echo "  - 相机帧: $$frame_num/$(FRAMES) ($(CAMERA_TYPE))"; \
		python3 main.py $(RENDER_OPTIONS) --frame $$frame_num --output "$$output_name"; \
	else \
		# 光源动画 (默认) \
		python3 main.py $(RENDER_OPTIONS) --light-frame $$frame_num --output "$$output_name"; \
	fi

# ===== 批量渲染目标 =====
frames: $(COLOR_FRAME_FILES)

# ===== 视频生成目标 =====
color_video: frames
	@echo "正在生成彩色视频..."
	@if [ -d "$(OUTPUT_DIR)/color" ] && [ $$(find "$(OUTPUT_DIR)/color" -maxdepth 1 -name 'frame_*.png' -print -quit | wc -l) -gt 0 ]; then \
		ffmpeg -y -framerate $(FPS) -start_number 0 -i "$(OUTPUT_DIR)/color/frame_%d.png" \
			-c:v libx264 -pix_fmt yuv420p -vf "scale=$(WIDTH):-2" $(COLOR_VIDEO) \
			&& echo "彩色视频已保存为: $(COLOR_VIDEO)" \
			|| { echo "错误: 彩色视频生成失败"; exit 1; }; \
	else \
		echo "错误: 没有找到彩色图像，无法生成视频"; \
		exit 1; \
	fi


depth_video: frames
ifeq ($(RENDER_DEPTH), 1)
	@echo "正在生成深度视频..."
	@if [ -d "$(OUTPUT_DIR)/depth" ] && [ $$(find "$(OUTPUT_DIR)/depth" -maxdepth 1 -name 'frame_*.png' -print -quit | wc -l) -gt 0 ]; then \
		ffmpeg -y -framerate $(FPS) -start_number 0 -i "$(OUTPUT_DIR)/depth/frame_%d.png" \
			-c:v libx264 -pix_fmt yuv420p -vf "scale=$(WIDTH):-2" $(DEPTH_VIDEO) \
			&& echo "深度视频已保存为: $(DEPTH_VIDEO)" \
			|| { echo "错误: 深度视频生成失败"; exit 1; }; \
	else \
		echo "警告: 没有找到深度图像，跳过深度视频生成"; \
	fi
else
	@echo "深度渲染已禁用，跳过深度视频生成"
endif

# ===== 单文件渲染目标 =====
render_single:
	@echo "渲染单个图像 (角度45°)..."
	@mkdir -p $(OUTPUT_DIR)/color $(OUTPUT_DIR)/depth
	@python3 main.py $(RENDER_OPTIONS) \
		--angle 45 \
		--total-frames 1 \
		--output "$(MODEL_NAME)_single"
	@echo "渲染完成: $(OUTPUT_DIR)/color/$(MODEL_NAME)_single.png"

# ===== 光源动画预设 =====
# 定义宏以简化光源动画目标的创建
define run_light_animation
	@echo "配置$(1)光源动画..."
	@$(MAKE) clean
	@$(MAKE) all \
		LIGHT_TYPE=point \
		LIGHT_ANIMATION=$(1) \
		LIGHT_RANGE=$(2) \
		ANIMATION_TYPE=light \
		$(3)
endef

light_vertical:
	$(call run_light_animation,vertical,1.0)

light_horizontal:
	$(call run_light_animation,horizontal,1.0)

light_circular:
	$(call run_light_animation,circular,1.0)

light_pulse:
	$(call run_light_animation,pulse,0.5)

light_figure8:
	$(call run_light_animation,figure8,1.0)

light_spiral:
	$(call run_light_animation,spiral,1.0)

light_custom:
	@echo "提示: 请使用单引号传递表达式，如 CUSTOM_X_EXPR='sin(2*pi*t)'"
	$(call run_light_animation,custom,$(LIGHT_RANGE), CUSTOM_X_EXPR='$(CUSTOM_X_EXPR)' CUSTOM_Y_EXPR='$(CUSTOM_Y_EXPR)' CUSTOM_Z_EXPR='$(CUSTOM_Z_EXPR)')

light_custom_fixed:
	$(call run_light_animation,custom,1.5, CUSTOM_X_EXPR='cos(4*pi*t)' CUSTOM_Y_EXPR='sin(4*pi*t)' CUSTOM_Z_EXPR='t*2-1' FRAMES=300 FPS=60)

# ===== 相机动画预设 (新增) =====
# 定义宏以简化相机动画目标的创建
define run_camera_animation
	@echo "配置$(1)相机动画..."
	@$(MAKE) clean
	@$(MAKE) all \
		ANIMATION_TYPE=camera \
		CAMERA_TYPE=$(1) \
		FRAMES=$(2) \
		FPS=$(3) \
		MODEL_NAME=$(MODEL_NAME)_cam_$(1) \
		$(4)
endef

# 左右摇头动画 (±30度)
camera_yaw:
	$(call run_camera_animation,yaw,90,30)

# 抬头低头动画 (±30度)
camera_pitch:
	$(call run_camera_animation,pitch,90,30)

# 歪头动画 (±30度)
camera_roll:
	$(call run_camera_animation,roll,90,30)

# 环绕物体旋转动画 (360度)
camera_orbit:
	$(call run_camera_animation,orbit,120,30)

# 一次生成全部四种相机动画
camera_all: camera_yaw camera_pitch camera_roll camera_orbit
	@echo "已完成所有相机动画"

# ===== 清理目标 =====
clean:
	@echo "清理输出目录..."
	rm -rf $(OUTPUT_DIR)
	@echo "清理完成"

clean_videos:
	@echo "清理视频文件..."
	rm -f $(COLOR_VIDEO) $(DEPTH_VIDEO)
	@echo "视频清理完成"

# ===== 帮助信息 =====
help:
	@echo "================== 3D模型渲染器使用说明 =================="
	@echo "使用方法:"
	@echo "  make				- 渲染所有帧并生成彩色和深度视频 (默认: $(ANIMATION_TYPE)动画)"
	@echo "  make color_only	 - 仅生成彩色视频"
	@echo "  make depth_only	 - 仅生成深度视频 (如果RENDER_DEPTH=1)"
	@echo "  make frames		 - 仅渲染所有帧但不生成视频"
	@echo "  make render_single  - 渲染单张图像(角度45度)"
	@echo "  make -j<N>		  - 使用N个并行进程加速渲染 (默认: $(NPROC))"
	@echo "  make clean		  - 删除输出目录 ($(OUTPUT_DIR))"
	@echo "  make clean_videos   - 删除生成的视频文件"
	@echo ""
	@echo "动画类型:"
	@echo "  make ANIMATION_TYPE=model  - 旋转模型 (默认)"
	@echo "  make ANIMATION_TYPE=light  - 移动光源 (使用光源动画预设)"
	@echo "  make ANIMATION_TYPE=camera - 相机动画 (使用相机动画预设)"
	@echo ""
	@echo "点光源动画预设 (自动设置ANIMATION_TYPE=light, COLORIZE=0, RENDER_DEPTH=0):"
	@echo "  make light_vertical   - 点光源上下移动动画"
	@echo "  make light_horizontal - 点光源左右移动动画"
	@echo "  make light_circular   - 点光源环绕移动动画"
	@echo "  make light_pulse	  - 点光源脉冲效果动画"
	@echo "  make light_figure8	- 点光源8字形轨迹动画"
	@echo "  make light_spiral	 - 点光源螺旋轨迹动画"
	@echo "  make light_custom [CUSTOM_X_EXPR='expr'] [CUSTOM_Y_EXPR='expr'] [CUSTOM_Z_EXPR='expr']"
	@echo "						- 自定义光源轨迹 (使用单引号!)"
	@echo "  make light_custom_fixed - 示例固定自定义轨迹"
	@echo ""
	@echo "相机动画预设 (自动设置ANIMATION_TYPE=camera):"
	@echo "  make camera_yaw	 - 相机左右摇头动画"
	@echo "  make camera_pitch   - 相机抬头低头动画"
	@echo "  make camera_roll	- 相机歪头动画"
	@echo "  make camera_orbit   - 相机环绕物体动画"
	@echo "  make camera_all	 - 生成所有四种相机动画"
	@echo ""
	@echo "通过命令行自定义参数:"
	@echo "  make OBJ_FILE=models/teapot.obj WIDTH=1920 HEIGHT=1080 FRAMES=240"
	@echo "  make PROJECTION=perspective FOCAL=1.5"
	@echo "  make USE_TEXTURE=1 TEXTURE=path/to/tex.png USE_MATERIALS=1"
	@echo "  make USE_LIGHTING=0"
	@echo "  make LIGHT_TYPE=directional LIGHT_DIR=0,-1,0"
	@echo "  make CAMERA_FROM=0,0,4 CAMERA_TYPE=orbit ANIMATION_TYPE=camera"
	@echo "  make RENDER_DEPTH=0 ZBUFFER=0"
	@echo ""
	@echo "当前配置:"
	@echo "  OBJ_FILE:		$(OBJ_FILE)"
	@echo "  MODEL_NAME:	  $(MODEL_NAME)"
	@echo "  OUTPUT_DIR:	  $(OUTPUT_DIR)"
	@echo "  分辨率:		 $(WIDTH)x$(HEIGHT)"
	@echo "  投影类型:	   $(PROJECTION) (焦距: $(FOCAL) 如果是透视投影)"
	@echo "  帧数:		   $(FRAMES) @ $(FPS) FPS"
	@echo "  动画类型:	   $(ANIMATION_TYPE)"
	@echo "  --- 相机 ---"
	@echo "  CAMERA_TYPE:	 $(CAMERA_TYPE)"
	@echo "  CAMERA_FROM:	 $(CAMERA_FROM)"
	@echo "  CAMERA_AT:	   $(CAMERA_AT)"
	@echo "  CAMERA_FOV:	  $(CAMERA_FOV)°"
	@echo "  --- 纹理 ---"
	@echo "  USE_TEXTURE:	 $(USE_TEXTURE) $(if $(filter 1,$(USE_TEXTURE)),(路径: $(TEXTURE), 类型: $(TEXTURE_TYPE), 大小: $(TEXTURE_SIZE)),)"
	@echo "  USE_MATERIALS:   $(USE_MATERIALS)"
	@echo "  COLORIZE:		$(COLORIZE)"
	@echo "  --- 光照 ($(if $(filter 1,$(USE_LIGHTING)),启用,禁用)) ---"
	@echo "  LIGHT_MODEL:	 $(LIGHT_MODEL)"
	@echo "  LIGHT_TYPE:	  $(LIGHT_TYPE)"
	@echo "  LIGHT_ANIMATION: $(LIGHT_ANIMATION) (范围: $(LIGHT_RANGE))"
	@echo "  --- 深度 ---"
	@echo "  RENDER_DEPTH:	$(RENDER_DEPTH)"
	@echo "  ZBUFFER:		 $(ZBUFFER)"
	@echo "  并行优化:	   检测到$(NPROC)个CPU核心，使用'-j$(NPROC)'"
	@echo "====================================================================="

# 防止 Make 将文件名误认为目标，保留中间生成文件
.SECONDARY: $(COLOR_FRAME_FILES) $(DEPTH_FRAME_FILES)