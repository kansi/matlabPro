function trainMatrix = reconstruct()
    testMatrix    = loadData('./dataset/1/');
    [~, trainMatrix] = loadData2('./CMU-PIE/');
%     trainM1  = loadData('./dataset/2/');
%     trainM2  = loadData('./dataset/3/');
%     trainM3  = loadData('./dataset/4/');
%     trainMatrix = [ trainM1 trainM2 trainM3 ];

    [~, tstColNo] = size(testMatrix);

    %calculate mean image
    [meanTrainImage, colNo] = calMean(trainMatrix);
    
    % subtract from mean matrix
    A     = trainMatrix - meanTrainImage(:, sum(eye(colNo)));
    testA = testMatrix - meanTrainImage(:, sum(eye(tstColNo)));
    
    %calculate eigen vectors for traning data
    eigenVec = calEigenVec(A, 25);
    
    [trainFeatureVec, eigenFace] = calFeatureVec(eigenVec, A);
    %testFeatureVec  = calFeatureVec(eigenVec, testA);
    
    [~, EFcolNo] = size(eigenFace);
    testFeatureVec = cell(1,tstColNo);
    for i=1:tstColNo
        omega = [];
        for j=1:EFcolNo
            omega = [ omega eigenFace(:,j)'*testA(:,i) ];
        end
        testFeatureVec{i}=omega';
    end 
    [~,m] = size(testFeatureVec);
    [~,n] = size(trainFeatureVec);
    
    image = [];
    for i=1:EFcolNo
        image = [ image eigenFace(:,i).*testFeatureVec{1}(i,1) ];
    end
    col_image = sum(image')' + meanTrainImage;
    img = reshape(col_image, [32, 32]);
    imshow(uint8(img))
    
    return
end

function [c, eigenFace] = calFeatureVec(eigenVec, A)
    [~,x] = size(eigenVec);
    [~,m] = size(A);
    c     = cell(1,m);
    
    eigenFace = A*eigenVec;
    %Normalization
    for i=1:x
        eigenFace(:,i) = eigenFace(:,i)/norm(eigenFace(:,i)); 
    end
    
    % calulate feature vector here
    for i=1:m
        omega = [];
        for j=1:x
            omega = [ omega  eigenFace(:,j)'*A(:,i) ];
        end
        c{i} = omega';
    end
end

function eigenVec = calEigenVec(A, x)
    prod    = A'*A;
    [V, D]  = eig(prod);             % Get eigen values of a'a
    eigenValues = diag(D);           % convert diagonal matric to 1D
    [~, index] = sort(eigenValues, 'descend'); % sort acc to eigenvalues
    % get x max eigen vectors
    newIndex = index(17:x+17);       % get the top x eigen values
    eigenVec = V(:, newIndex);       % get the corresponding eigenvectors
end

function [meanImage, cols] = calMean(matrix)
    [~,cols]  = size(matrix);
    mean      = sum(matrix');
    meanImage = (mean/cols)';
end

function imageMatrix = loadData(path)
    image = dir(strcat(path, '*.pgm'));
    [m,n] = size(image);
    tmp = [];                       % create an empty matrix
    % for loop for all images
    for i=1:m
        %disp(image(i).name)
        a = double(imread(strcat(path, image(i).name), 'pgm'));  % PxQ matrix of a single image
        a = imresize(a, [32 32]);
        [p,q] = size(a);            % get the dimensions of the matrix
        a = reshape(a', 1, p*q)';   % convert matrix to column
        tmp = [tmp a];              % concatenate the columns into the new matrix
    end
    imageMatrix = tmp;
end

function [testMatrix, trainMatrix] = loadData2(path)
    data = load(strcat(path, 'CMUPIEData.mat'));
    [~,m] = size(data);
    trainMatrix = [];            % create an empty matrix
    testMatrix  = [];            % create an empty matrix
    % for loop for all different labels
    %images = cell(1,68);
    for i=0:67
        range = data(1).CMUPIEData(1+42*i:20+42*i);
        trainMatrix = [ trainMatrix classify(range, 1,5)  ];
        trainMatrix = [ trainMatrix classify(range, 6,10) ];
        trainMatrix = [ trainMatrix classify(range, 11,15)];
        testMatrix =  [ testMatrix  classify(range, 16,20)];
    end
    %testMatrix
    %size(trainMatrix)
    return
end

function matrix = classify(structs,x,y)
    matrix = [];
    for i=x:y
        matrix = [ matrix structs(i).pixels'];
    end
    return
end
