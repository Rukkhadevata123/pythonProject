# 3D模型渲染器批处理系统 - 支持并行任务处理
# 功能：渲染OBJ模型，支持纹理和深度图生成

# ===== 基本渲染参数配置 =====
OBJ_FILE ?= bunny/bunny2k_f.obj
MODEL_NAME ?= $(basename $(notdir $(OBJ_FILE)))
WIDTH ?= 1200
HEIGHT ?= 1200
PROJECTION ?= orthographic
FRAMES ?= 120
OUTPUT_DIR ?= output_$(MODEL_NAME)
FPS ?= 30
COLOR_VIDEO ?= $(MODEL_NAME)_$(PROJECTION).mp4
DEPTH_VIDEO ?= $(MODEL_NAME)_$(PROJECTION)_depth.mp4

# ===== 渲染选项 =====
TEXTURE ?=
TEXTURE_TYPE ?= checkerboard
TEXTURE_SIZE ?= 512
USE_TEXTURE ?= 0
USE_MATERIALS ?= 1
COLORIZE ?= 1
USE_LIGHTING ?= 1       # 新增：是否使用光照
LIGHT_MODEL ?= phong    # 新增：光照模型类型
AMBIENT ?= 0.2          # 新增：环境光强度
DIFFUSE ?= 0.6          # 新增：漫反射强度
SPECULAR ?= 0.2         # 新增：高光强度
SHININESS ?= 32.0       # 新增：高光锐度
LIGHT_DIR ?= 1,-1,1     # 新增：光源方向
RENDER_DEPTH ?= 1
DEPTH_MIN ?= 1
DEPTH_MAX ?= 99
ZBUFFER ?= 1
FOCAL ?= 2.0

# ===== 并行处理配置 =====
NPROC = $(shell nproc 2>/dev/null || echo 4)
PARALLEL = -j$(NPROC)

# ===== 帧序列生成 =====
FRAME_NUMS := $(shell seq 0 $(shell expr $(FRAMES) - 1))
FRAME_FILES := $(foreach n,$(FRAME_NUMS),$(OUTPUT_DIR)/color/frame_$(shell printf "%03d" $(n)).png)
DEPTH_FILES := $(foreach n,$(FRAME_NUMS),$(OUTPUT_DIR)/depth/frame_$(shell printf "%03d" $(n)).png)

# ===== 渲染选项构建 =====
RENDER_OPTIONS = --obj $(OBJ_FILE) \
	--width $(WIDTH) --height $(HEIGHT) \
	--projection $(PROJECTION) \
	--focal $(FOCAL) \
	--output-dir $(OUTPUT_DIR) \
	--texture-type $(TEXTURE_TYPE) \
	--texture-size $(TEXTURE_SIZE) \
	--depth-min $(DEPTH_MIN) --depth-max $(DEPTH_MAX) \
	--light-model $(LIGHT_MODEL) \
	--ambient $(AMBIENT) \
	--diffuse $(DIFFUSE) \
	--specular $(SPECULAR) \
	--shininess $(SHININESS) \
	--light-dir $(LIGHT_DIR)

ifeq ($(USE_TEXTURE), 0)
	RENDER_OPTIONS += --no-texture
else
	ifneq ($(TEXTURE),)
		RENDER_OPTIONS += --texture $(TEXTURE)
	endif
endif

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

ifeq ($(USE_LIGHTING), 0)
	RENDER_OPTIONS += --no-lighting
endif

# ===== 主要目标 =====
all: color_video depth_video

color_only: color_video
depth_only: depth_video

# ===== 目录创建目标 =====
$(OUTPUT_DIR)/color $(OUTPUT_DIR)/depth:
	mkdir -p $@

# ===== 单帧渲染规则 =====
$(OUTPUT_DIR)/color/frame_%.png: | $(OUTPUT_DIR)/color $(OUTPUT_DIR)/depth
	@echo "渲染帧 $*..."
	@python3 main.py $(RENDER_OPTIONS) \
		--angle $(shell echo "$* * 360 / ($(FRAMES) - 1)" | bc -l) \
		--output "frame_$*"

# ===== 批量渲染目标 =====
frames: $(FRAME_FILES)

# ===== 视频生成目标 =====
color_video: frames
	@echo "正在生成彩色视频..."
	ffmpeg -y -framerate $(FPS) -pattern_type glob -i "$(OUTPUT_DIR)/color/frame_*.png" \
		-c:v libx264 -pix_fmt yuv420p -vf "scale=$(WIDTH):-2" $(COLOR_VIDEO)
	@echo "彩色视频已保存为: $(COLOR_VIDEO)"

depth_video: frames
ifeq ($(RENDER_DEPTH), 1)
	@echo "正在生成深度视频..."
	ffmpeg -y -framerate $(FPS) -pattern_type glob -i "$(OUTPUT_DIR)/depth/frame_*.png" \
		-c:v libx264 -pix_fmt yuv420p -vf "scale=$(WIDTH):-2" $(DEPTH_VIDEO)
	@echo "深度视频已保存为: $(DEPTH_VIDEO)"
else
	@echo "深度渲染已禁用，跳过深度视频生成"
endif

# ===== 单文件渲染目标 =====
render_single:
	@echo "渲染单个图像..."
	@python3 main.py $(RENDER_OPTIONS) \
		--angle 45 \
		--output "$(MODEL_NAME)_single"
	@echo "渲染完成：$(OUTPUT_DIR)/color/$(MODEL_NAME)_single.png"

# ===== 清理目标 =====
clean:
	@echo "清理临时文件..."
	rm -rf $(OUTPUT_DIR)
	@echo "清理完成"

clean_videos:
	rm -f $(COLOR_VIDEO) $(DEPTH_VIDEO)

# ===== 帮助信息 =====
help:
	@echo "================== 三角形渲染器使用说明 =================="
	@echo "使用方法:"
	@echo "  make				- 渲染所有帧并生成彩色和深度视频"
	@echo "  make color_only	 - 仅生成彩色视频"
	@echo "  make depth_only	 - 仅生成深度视频"
	@echo "  make frames		 - 仅渲染所有帧但不生成视频"
	@echo "  make render_single  - 渲染单张图像(角度45度)"
	@echo "  make -j8			- 使用8个并行进程加速渲染"
	@echo "  make clean		  - 删除生成的帧图像"
	@echo "  make clean_videos   - 删除生成的视频文件"
	@echo ""
	@echo "自定义参数示例:"
	@echo "  make OBJ_FILE=models/teapot.obj PROJECTION=perspective"
	@echo "  make TEXTURE=my_texture.png USE_MATERIALS=1"
	@echo ""
	@echo "基本参数:"
	@echo "  OBJ_FILE = $(OBJ_FILE)"
	@echo "  WIDTH = $(WIDTH)"
	@echo "  HEIGHT = $(HEIGHT)"
	@echo "  PROJECTION = $(PROJECTION)"
	@echo "  FRAMES = $(FRAMES)"
	@echo "  FPS = $(FPS)"
	@echo "  OUTPUT_DIR = $(OUTPUT_DIR)"
	@echo "  FOCAL = $(FOCAL)"
	@echo ""
	@echo "纹理选项:"
	@echo "  TEXTURE = $(TEXTURE)"
	@echo "  TEXTURE_TYPE = $(TEXTURE_TYPE)"
	@echo "  TEXTURE_SIZE = $(TEXTURE_SIZE)"
	@echo "  USE_TEXTURE = $(USE_TEXTURE)"
	@echo ""
	@echo "材质和颜色选项:"
	@echo "  USE_MATERIALS = $(USE_MATERIALS)"
	@echo "  COLORIZE = $(COLORIZE)"
	@echo ""
	@echo "深度图选项:"
	@echo "  RENDER_DEPTH = $(RENDER_DEPTH)"
	@echo "  ZBUFFER = $(ZBUFFER)"
	@echo "  DEPTH_MIN = $(DEPTH_MIN)"
	@echo "  DEPTH_MAX = $(DEPTH_MAX)"
	@echo ""
	@echo "当前并行配置: 检测到$(NPROC)个CPU核心，使用$(PARALLEL)选项"

.PHONY: all color_only depth_only frames color_video depth_video render_single clean clean_videos help
