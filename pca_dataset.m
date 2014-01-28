function trainMatrix = pca_dataset()
    trainM1     = loadData('./dataset/1/');
    trainM2     = loadData('./dataset/2/');
    trainM3     = loadData('./dataset/3/');
    testMatrix  = loadData('./dataset/4/');
    trainMatrix = [];
    for i=0:37
        trainMatrix = [ trainMatrix trainM1(:,1+5*i:(i+1)*5)];
        trainMatrix = [ trainMatrix trainM2(:,1+5*i:(i+1)*5)];
        trainMatrix = [ trainMatrix trainM3(:,1+5*i:(i+1)*5)];
    end

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
    
    %%%Classify the images
    success = 0;
    for i=1:m
        testImage = testFeatureVec{i}(:,1);
        mag       = 9999999999999999999999999999999999999999;
        match     = [0,0];
        for j=1:n
            diff = norm(testImage - trainFeatureVec{j}(:,1));
            if diff < mag
                mag       = diff;
                match(:,1)=i;
                match(:,2)=j;
            end
        end
        %match
        if match(:,1)/5 == 0
            match(:,1) = match(:,1)-1;
        end        
        testClass  = ceil(match(:,1)/5);
        if match(:,2)/15 == 0
            match(:,2) = match(:,2)-1;
        end
        trainClass = ceil(match(:,2)/15);
        %match = [testClass, trainClass]
        if testClass == trainClass
            success = success + 1;
        end
    end
    accuracy = success*100/m
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
    tmp   = [];                       % create an empty matrix
    % for loop for all images
    for i=1:m
        %disp(image(i).name)
        a = double(imread(strcat(path, image(i).name), 'pgm'));  % PxQ matrix of a single image
        [p,q] = size(a);            % get the dimensions of the matrix
        a = reshape(a', 1, p*q)';   % convert matrix to column
        tmp = [tmp a];              % concatenate the columns into the new matrix
    end
    imageMatrix = tmp;
end
