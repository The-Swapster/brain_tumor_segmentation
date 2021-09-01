
img = imread('/Downloads/1389961.jpg');

%% Filter

disp('Preprocessing image please wait . . .');
inp = imdiffusefilt(img); %applying anisotropic filter
inp = uint8(inp);
inp = imresize(inp,[256,256]);
if size(inp,3) > 1
    inp = rgb2gray(inp);
end
% figure();
% imshow(inp);
% title('Filtered image');

%% thresholding
sout = imresize(inp, [256,256]);
t0 = 60;
th = t0 + ((max(inp(:)) + min(inp(:)))/2);
for i = 1:1:size(inp,1)
    for j = 1:1:size(inp,2)
        if inp(i,j) > th
            sout(i,j) = 1;
        else
            sout(i,j) = 0;
        end
    end
end

%% Morphological Operation

label = bwlabel(sout);% returns label matrix that contains 8 connected components found in image
stats = regionprops(logical(sout), 'Solidity', 'Area', 'BoundingBox');%returns measurements for the set of properties for each 8-connected component in the input image 
density = [stats.Solidity];
area = [stats.Area];
high_dense_area = (1>density) & (density>0.0);
max_area = max(area(high_dense_area));
tumor_label = find(area == max_area);
tumor = ismember(label, tumor_label);

if max_area > 100
%     figure();
%     imshow(tumor)
%     title('Tumor');
else
    h = msgbox('Normal!!','status');
    disp('normal');
    return;
end

%% Bounding box

box = stats(tumor_label);
wantedBox = box.BoundingBox;
% figure();
% imshow(inp);
% title('Bounding Box');
% hold on;
% rectangle('Position',wantedBox,'EdgeColor','y');
% hold off;

%% Getting Tumor Outline - image filling, eroding, subtracting
% erosion the walls by a few pixels

% dilationAmount = 5;
% rad = floor(dilationAmount);
% [r,c] = size(tumor);
% filledImage = imfill(tumor, 'holes');
% 
% for i=1:r
%     for j=1:c
%         x1=i-rad;
%         x2=i+rad;
%         y1=1-rad; % should be j-rad, which would give a thick background
%         y2=j+rad;
%         if x1<1
%             x1=1;
%         end
%         if x2>r
%             x2=r;
%         end
%         if y1<1
%             y1=1;
%         end
%         if y2>c
%             y2=c;
%         end
%         erodedImage(i,j) = min(min(filledImage(x1:x2,y1:y2)));
%     end
% end
% figure();
% imshow(erodedImage);
% title('eroded image');

%% subtracting eroded image from original BW image

% tumorOutline = tumor;
% tumorOutline(erodedImage) = 0;

%% Inserting the outline in filtered image in red color

rgb = inp(:,:,[1 1 1]);
red = rgb(:,:,1);
red(tumor) = 255;
green = rgb(:,:,2);
green(tumor) = 0;
blue = rgb(:,:,3);
blue(tumor) = 0;

tumorOutlineInserted(:,:,1) = red;
tumorOutlineInserted(:,:,2) = green;
tumorOutlineInserted(:,:,3) = blue;

% figure();
% imshow(tumorOutlineInserted);
% title('Detected Tumer');

%% Displayed Together

figure();
subplot(2,2,1);imshow(img);title('Input image');

subplot(2,2,2);imshow(inp);title('Bounding Box');
hold on;rectangle('Position',wantedBox,'EdgeColor','y');hold off;

subplot(2,2,3);imshow(tumor);title('tumor alone');
subplot(2,2,4);imshow(tumorOutlineInserted);title('Detected Tumor');
