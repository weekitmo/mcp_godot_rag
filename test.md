# ImageMagick 是一个功能强大的图片处理工具，以下是一些常用的处理操作：

1. 基础图片操作：
   - 调整大小：`magick input.jpg -resize 800x600 output.jpg`
   - 裁剪：`magick input.jpg -crop 300x300+100+100 output.jpg`
   - 旋转：`magick input.jpg -rotate 90 output.jpg`

2. 图片质量和格式：
   - 压缩质量：`magick input.jpg -quality 80 output.jpg`
   - 格式转换：`magick input.png output.jpg`
   - 优化大小：`magick input.jpg -strip output.jpg`

3. 图片效果：
   - 模糊：`magick input.jpg -blur 0x8 output.jpg`
   - 锐化：`magick input.jpg -sharpen 0x3 output.jpg`
   - 亮度/对比度：`magick input.jpg -brightness-contrast 10x5 output.jpg`
   - 灰度：`magick input.jpg -grayscale average output.jpg`

4. 图片合成：
   - 添加边框：`magick input.jpg -border 10 output.jpg`
   - 添加水印：`magick input.jpg watermark.png -composite output.jpg`
   - 图片拼接：`magick image1.jpg image2.jpg +append output.jpg`

5. 批量处理：
   - 处理目录中所有图片：`magick mogrify -resize 800x600 *.jpg`
   - 批量格式转换：`magick mogrify -format png *.jpg`

6. 特殊效果：
   - 油画效果：`magick input.jpg -paint 3 output.jpg`
   - 素描效果：`magick input.jpg -charcoal 2 output.jpg`
   - 浮雕效果：`magick input.jpg -emboss 2 output.jpg`

7. 图片信息：
   - 查看图片信息：`magick identify input.jpg`
   - 显示详细信息：`magick identify -verbose input.jpg`

8. 颜色处理：
   - 颜色平衡：`magick input.jpg -modulate 100,150,100 output.jpg`
   - 色彩饱和度：`magick input.jpg -saturation 150 output.jpg`
   - 自动色彩：`magick input.jpg -auto-level output.jpg`